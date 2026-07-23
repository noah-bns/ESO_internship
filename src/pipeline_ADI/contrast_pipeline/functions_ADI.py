
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import os, re
from multiprocessing import cpu_count
from scipy import interpolate
import importlib
import shutil
from typing import Tuple, Callable, Optional
from typing import List, Dict, Union
import pandas as pd

#images
import torch
import imageio.v2 as imageio
from astropy.io import fits
from astropy.visualization import LogStretch, ImageNormalize, AsinhStretch
from astropy.modeling import models, fitting
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass
import matplotlib.gridspec as gridspec
import seaborn as sns

#scientific libraries
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, fpf_2_gaussian_sigma, LaplaceBootstrapTest
from fours.detection_limits.applefy_wrapper import CADIDataReductionGPU #, PCADataReductionGPU
from .pca_utils import PCADataReductionGPU
from applefy.detections.contrast import Contrast
from applefy_extensions.contrast_curves import ContrastFast



import numpy as np
from skimage.registration import phase_cross_correlation


def estimate_center(stack, nframes=1000, upsample_factor=100,
                    max_iter=100, tol=1e-3):
    """
    Estimate the center of a stack from the median of the first nframes
    using iterative 180° rotational phase cross-correlation.

    Returns
    -------
    yc, xc : float
        Subpixel center coordinates.
    """

    ref = np.median(stack[:nframes], axis=0)

    ny, nx = ref.shape
    yc = (ny - 1) / 2
    xc = (nx - 1) / 2

    for _ in range(max_iter):

        half = int(min(
            yc,
            xc,
            ny - 1 - yc,
            nx - 1 - xc
        ))

        y = int(round(yc))
        x = int(round(xc))

        sub = ref[y-half:y+half+1,
                  x-half:x+half+1]

        rot = np.rot90(sub, 2)

        shift, _, _ = phase_cross_correlation(
            sub,
            rot,
            upsample_factor=upsample_factor
        )

        dy, dx = shift / 2

        yc += dy
        xc += dx

        if np.hypot(dy, dx) < tol:
            print(f"Center converged after {_+1} iterations.")
            break

    return (yc, xc)

def crop_stack(stack, 
               coords = None,
               radius = None
               ):
    """
    Crop every frame to the largest odd square centered on (yc, xc).
    """

    if coords is None:
        yc, xc = estimate_center(stack)
    
    else:
        yc, xc = coords
    
    ny, nx = stack.shape[1:]

    if radius is None:
        half = int(min(
            yc,
            xc,
            ny - 1 - yc,
            nx - 1 - xc
        ))
    else:
        half = radius
        
    yc = int(round(yc))
    xc = int(round(xc))

    return stack[
        :,
        yc-half:yc+half+1,
        xc-half:xc+half+1
    ]


def zoom_to_peak(
        img, 
        radius,
        method = 'com'
        ):
    """
    Zoom into the peak of the PSF image.
    method: 
    - 'com' for center of mass, https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.ndimage.center_of_mass.html
    - 'max' for maximum pixel value
    """
#     if method == 'com':
#         coords_maxpsf = (np.round(np.array(center_of_mass(img))).astype(int))
#     if method == 'max':
#         coords_maxpsf = (np.unravel_index(np.argmax(img, axis=None), img.shape))

#     img = img[coords_maxpsf[0]-radius:coords_maxpsf[0]+radius+1, coords_maxpsf[1]-radius:coords_maxpsf[1]+radius+1]
#     return img

    if img.ndim == 2:
        ref = img
    elif img.ndim == 3:
        ref = np.median(img[:10], axis=0) # bcs already centred, not np.median(img, axis=0)   # or np.mean(img, axis=0)
    else:
        raise ValueError("img stack must be 2D or 3D")

    if method == "com":
        y, x = np.round(center_of_mass(ref)).astype(int)
    elif method == "max":
        y, x = np.unravel_index(np.argmax(ref), ref.shape)
    else:
        raise ValueError("Unknown method")

    img = img[
        ...,
        y - radius : y + radius + 1,
        x - radius : x + radius + 1,
    ]

    return img


def calculate_fwhm(
        psf):
    y, x = np.mgrid[:psf.shape[0], :psf.shape[1]]

    # Initial Gaussian guess
    g_init = models.Gaussian2D(
        amplitude=psf.max(),
        x_mean=psf.shape[1] / 2,
        y_mean=psf.shape[0] / 2,
        x_stddev=3,
        y_stddev=3
    )

    # Fit
    fit_p = fitting.LevMarLSQFitter()
    g_fit = fit_p(g_init, x, y, psf)

    # Convert sigma -> FWHM
    fwhm_x = 2.355 * g_fit.x_stddev.value
    fwhm_y = 2.355 * g_fit.y_stddev.value

    print(f"FWHM_x = {fwhm_x:.2f} pix")
    print(f"FWHM_y = {fwhm_y:.2f} pix")

    # Often people use the mean:
    fwhm = 0.5 * (fwhm_x + fwhm_y)

    print(f"Mean FWHM = {fwhm:.2f} pix \n")
    return fwhm
    

def fake_planet_experiment(
    contrast_instance,
    output_path: Path,
    dataset: dict,
    fp_config: dict,
    separations: np.ndarray,
    algo_name: str,
    angles: np.ndarray,
    #grid: bool
) -> object:
    """
    Run fake planet injection experiment with config-driven parameters.
    
    Parameters
    ----------
    contrast_instance : Contrast
        Contrast analysis instance.
    dataset : dict
        Processed dataset dictionary.
    fp_config : dict
        Fake planet configuration dictionary.
    separations : np.ndarray
        Separation values in pixels.
    algo_name : str
        Algorithm version ('PCAD', 'CADI', etc.).
    angles : np.ndarray
        Position angles in degrees.

    Returns
    -------
    Contrast
        Updated contrast instance.
    """

    # # Remove existing directory and all its contents
    # if output_path.exists():
    #     shutil.rmtree(output_path)
    #     print(f"Removed existing directory to avoid overwrites: {output_path}.")

    # output_path.mkdir(
    #     parents=True,
    #     exist_ok=True
    # )

    # flux_ratio = fp_config['flux_ratio_mag']
    # if isinstance(flux_ratio, list):
    #     flux_ratio = np.array(flux_ratio)
    # flux_ratio = mag2flux_ratio(flux_ratio)

    # contrast_instance = ContrastFast(
    #     science_sequence=dataset["sci_img"],
    #     psf_template=dataset["psf"],
    #     parang_rad=angles,
    #     psf_fwhm_radius=dataset["fwhm"] / 2,
    #     dit_psf_template=dataset["dit_psf"],
    #     device = fp_config['device'],
    #     dit_science=dataset["dit_science"],
    #     scaling_factor=fp_config["scaling_factor"],
    #     checkpoint_dir= output_path
    # )


    # contrast_instance.design_fake_planet_experiments(
    #     flux_ratios= flux_ratio,
    #     num_planets=fp_config['num_fake_planets'],
    #     separations = separations,
    #     overwrite=True,
    #     )

    # num_parallel = cpu_count()//2

    if algo_name == 'PCAD':

        work_dir = contrast_instance.scratch_dir / Path("tensorboard_pca")


        algorithm_function = PCADataReductionGPU(
            pca_numbers=fp_config['components'],
            device=fp_config['device'],
            pca_method=fp_config['pca_method'],
            niter=fp_config['niter'],
            random_state=fp_config['random_state'],
            eps=fp_config['eps'],
            work_dir = work_dir,
            approx_svd_trunc=fp_config['approx_svd_trunc'],
            subsample_rotation_grid=fp_config['subsample_rotation_grid'],
            combine=fp_config['combine'],
        )
            
    if algo_name == 'CADI':
    
        algorithm_function = CADIDataReductionGPU(
            device = fp_config['device']
        )
        
    # try:
    #     contrast_instance.run_fake_planet_experiments(
    #         algorithm_function=algorithm_function,
    #         num_parallel=num_parallel)
    # except:
        # can fail in multiprocessing, depending on whether optional dependencies are installed or not
    num_parallel=1
    contrast_instance.run_fake_planet_experiments(
        algorithm_function=algorithm_function,
        num_parallel=num_parallel)
                

    return contrast_instance


def compute_contrast(
        contrast_instance, 
        fwhm, 
        pixel_scale, 
        grid,
        photometry = 'FS', 
        test = 't-test',
        ):
    """
    Compute analytic contrast curves for a processed high-contrast imaging
    dataset using the Appleby tutorial workflow.
    see: https://applefy.readthedocs.io/en/latest/02_user_documentation/01_contrast_curves.html

    The function configures the photometric extraction strategy and
    statistical test, prepares the contrast analysis products, and computes
    contrast curves with associated uncertainties.

    Parameters
    ----------
    contrast_instance : object
        Contrast analysis instance containing the processed dataset and
        fake planet experiment results.
    fwhm : float
        Full width at half maximum (FWHM) of the PSF in pixels.
    photometry : {'FS', 'AS'}, optional
        Photometry extraction method.
        - ``'FS'`` : spaced pixel sampling (default)
        - ``'AS'`` : aperture-sum photometry
    test : {'t-test', 'bootstrap'}, optional
        Statistical test used to estimate detection significance.
        - ``'t-test'`` : assumes Gaussian residual noise (default)
        - ``'bootstrap'`` : Laplacian residual noise model following
          Bonse et al. (2023)
    pixel_scale : float
        Pixel size in arcseconds.
    grid : bool, optional
        If True, compute contrast curves on a grid of separations and flux
        ratios. If False, compute contrast curves at specified separations
        only (default: False).

    Returns
    -------
    if grid is False:
        contrast_curves : object
            Computed analytic contrast curves.
        contrast_errors : object
            Uncertainties associated with the contrast curves.
    if grid is True:
        contrast_curves_grid : dict
            Contrast grids (one for each number of PCA components)
        contrast_grids : pandas DataFrame
            Contrast curves obtained by thresholding the contrast grids.
    
    Warns
    -----
    UserWarning
        Raised if an unsupported photometry mode or statistical test is
        provided.
    """

    if photometry == 'FS':# Use spaced pixel values
        photometry_mode_planet = AperturePhotometryMode(
            "F", # or "P"
            psf_fwhm_radius=fwhm/2,
            search_area=0.5)
        photometry_mode_noise = AperturePhotometryMode(
            "P",
            psf_fwhm_radius=fwhm/2)    
    elif photometry == 'AS': # Use apertures pixel values
        photometry_mode_planet = AperturePhotometryMode(
            "ASS", # or "AS"
            psf_fwhm_radius=fwhm/2,
            search_area=0.5)

        photometry_mode_noise = AperturePhotometryMode(
            "AS",
            psf_fwhm_radius=fwhm/2)
    else:
        # Issue a warning
        warnings.warn("Photometry style not recognized, 'FS' for spaced pixel values and 'AS' for aperture sums.", UserWarning)

    contrast_instance.prepare_contrast_results(
        photometry_mode_planet=photometry_mode_planet,
        photometry_mode_noise=photometry_mode_noise)
    
    if test == 't-test':
        statistical_test = TTest()
    elif test == 'bootstrap':
        #The Parametric Bootstrap test as discussed in (Bonse et al. 2023) (assumes Laplacian residual noise). You can download the lookup from Zenodo.
        statistical_test = LaplaceBootstrapTest.construct_from_json_file("file/to/lookup_table")
    else:
        # Issue a warning
        warnings.warn("Statistical testing style not recognized, 't-test' assumes gaussian residual noise, and 'bootstrap' assumes Laplacian residual noise (you can download the lookup from Zenodo).", UserWarning)

    if grid ==False:
        contrast_curves, contrast_errors = contrast_instance.compute_analytic_contrast_curves(
            statistical_test=statistical_test,
            confidence_level_fpf=gaussian_sigma_2_fpf(5),
            num_rot_iter=20,
            pixel_scale= pixel_scale)

        return contrast_curves, contrast_errors
    
    if grid == True:
        contrast_curves_grid, contrast_grids = contrast_instance.compute_contrast_grids(
            statistical_test=statistical_test,
            confidence_level_fpf=gaussian_sigma_2_fpf(5),
            num_rot_iter=10,
            safety_margin=2.5,
            num_cores=45, 
            pixel_scale= pixel_scale)

        return contrast_curves_grid, contrast_grids


def plot_contrast_grid(
    contrast_grid_axis,
    colorbar_axis,
    contrast_grid,
    cmap = "YlGnBu",
    vmax=2, 
    vmin=7):

    c_bar_kargs = dict(
        orientation = "vertical",
        label = r"Confidence [$\sigma_{\mathcal{N}}$]")

    heat = sns.heatmap(
        contrast_grid,
        vmax=vmax, vmin=vmin,
        annot=True,
        cmap= cmap,
        fmt = '.1f',
        ax=contrast_grid_axis,
        cbar_ax=colorbar_axis,
        cbar_kws=c_bar_kargs)

    ylabels = ['{:.1f}'.format(float(x.get_text()))
               for x in heat.get_yticklabels()]
    _=heat.set_yticklabels(ylabels)
    xlabels = ['{:.1f}'.format(float(x.get_text()))
               for x in heat.get_xticklabels()]
    _=heat.set_xticklabels(xlabels)


def plot_contrast_curves(
    contrast_curves, 
    lim_mag_y,
    curves_output_path, 
    dataset_name,
    contrast_errors = None, 
    lim_x = None, 
    title = None,
    cmap = "magma",
    alpha = 0.7,
    ):

    # compute the overall best contrast curve
    PADI_values = contrast_curves.loc[:, ~contrast_curves.columns.str.contains("cADI", regex=True)]
    if len(PADI_values.columns) > 0:
        overall_best = np.min(PADI_values.values, axis=1)

        # get the error bars of the the overall best contrast curve
        best_idx = np.argmin(PADI_values.values, axis=1)

        # Find one color for each number of PCA components used
        color_map = plt.cm.get_cmap(cmap)   # seaborn-style colormap available in mpl
        colors = [color_map(int(i)) for i in np.round(np.linspace(0, 220, len(PADI_values.columns)))]

    if isinstance(contrast_curves.index, pd.MultiIndex):
        separations_arcsec = contrast_curves.index.get_level_values(0)
        separations_FWHM = contrast_curves.index.get_level_values(1)

        x = separations_FWHM        # bottom axis
        x_top = separations_arcsec  # top axis
        multi_index = True
    else:
        x = contrast_curves.index
        multi_index = False

    # 1.) Create Plot Layout
    fig = plt.figure(constrained_layout=False, figsize=(12, 8))
    gs0 = fig.add_gridspec(1, 1)
    axis_contrast_curves = fig.add_subplot(gs0[0, 0])


    # ---------------------- Create the Plot --------------------
    i = 0 # color picker
    for tmp_model in contrast_curves.columns:
        
        if 'cADI'.lower() in tmp_model.lower():
            num_components = 'cADI'
            color = 'red'
        else:
            num_components = int(tmp_model[5:8])
            color = colors[i]
            i+=1
        tmp_flux_ratios = contrast_curves.reset_index(
            level=0)[tmp_model].values


        axis_contrast_curves.plot(
            x,
            tmp_flux_ratios,
            color = color,
            alpha = alpha,
            label=num_components)

        if contrast_errors is not None:
            tmp_errors = contrast_errors.reset_index(
                level=0)[tmp_model].values
            
            axis_contrast_curves.fill_between(
                x,
                tmp_flux_ratios + tmp_errors,
                tmp_flux_ratios - tmp_errors,
                color = color,
                alpha=alpha/2)
            
            if len(PADI_values.columns) > 0:
                best_contrast_errors = contrast_errors.values[np.arange(len(best_idx)), best_idx]

                axis_contrast_curves.fill_between(
                    x,
                    overall_best + best_contrast_errors,
                    overall_best - best_contrast_errors,
                    color = 'blue',
                    alpha=alpha/2)

    axis_contrast_curves.set_yscale("log")
    # ------------ Plot the overall best -------------------------
    if len(PADI_values.columns) > 0:
        axis_contrast_curves.plot(
            x,
            overall_best,
            color = "blue",
            lw=3,
            ls="--",
            label="Best")

        best_pca = [
            int(re.search(r"PCA_(\d+)", col).group(1))
            for col in PADI_values.columns[best_idx]
        ]
        #save the overall best
        result = pd.DataFrame(
            {
                #"separation_FWHM": x,
                "best_contrast": overall_best,
                "best_PCA": best_pca,
            },
            index=PADI_values.index,
        )

    # ------------- Double axis and limits -----------------------
    if lim_x:
        lim_x = lim_x
    else:
        lim_x = (np.min(x) - 0.01, np.max(x) + 0.01)


    axis_contrast_curves_mag = axis_contrast_curves.twinx()
    axis_contrast_curves_mag.plot(
        x,
        flux_ratio2mag(tmp_flux_ratios),
        alpha=0.)
    axis_contrast_curves_mag.invert_yaxis()


    axis_contrast_curves.grid(which='both')
    axis_contrast_curves_mag.set_ylim(*lim_mag_y)
    axis_contrast_curves.set_ylim(
        (mag2flux_ratio(lim_mag_y[0])),
        (mag2flux_ratio(lim_mag_y[1])))

    axis_contrast_curves.set_xlim(*lim_x)
    axis_contrast_curves_mag.set_xlim(*lim_x)


    if multi_index:
        fwhm_to_arcsec = interpolate.interp1d(
            separations_FWHM,
            separations_arcsec,
            fill_value="extrapolate"
        )

        axis_contrast_curves_arcsec = axis_contrast_curves.twiny()
        axis_contrast_curves_arcsec.plot(
            separations_arcsec,
            tmp_flux_ratios,
            alpha=0,
        )

        axis_contrast_curves_arcsec.set_xlim(
            *fwhm_to_arcsec(lim_x)
        )
    # ----------- Labels and fontsizes --------------------------
    
        axis_contrast_curves_arcsec.set_xlabel(
            "Separation [arcsec]", size=16
        )
        axis_contrast_curves_arcsec.tick_params(
            axis='both', which='major', labelsize=14)
    
        axis_contrast_curves.set_xlabel(
            "Separation [FWHM]", size=16
        )
    else:
        axis_contrast_curves.set_xlabel(
            "Separation [FWHM]", size=16
        )

    axis_contrast_curves.set_ylabel(
        r"Planet-to-star flux ratio", size=16)
    axis_contrast_curves_mag.set_ylabel(
        r"$\Delta$ Magnitude", size=16)

    axis_contrast_curves.tick_params(
        axis='both', which='major', labelsize=14)

    axis_contrast_curves_mag.tick_params(
        axis='both', which='major', labelsize=14)

    # # ----------- Labels and fontsizes --------------------------
    if title:
        set_title = title
    else:
        set_title = r"$5 \sigma_{\mathcal{N}}$ Contrast Curves"
    axis_contrast_curves_mag.set_title(
        set_title,
        fontsize=18, fontweight="bold", y=1.05)

    # --------------------------- Legend -----------------------
    handles, labels = axis_contrast_curves.\
        get_legend_handles_labels()

    leg1 = fig.legend(handles, labels,
                    bbox_to_anchor=(0.12, -0.1),
                    fontsize=14,
                    title="# PCA components",
                    loc='lower left', ncol=8)

    _=plt.setp(leg1.get_title(),fontsize=14)
    plt.savefig(f"{(curves_output_path)}/Contrast_Curves_{dataset_name}.png", pad_inches=0.15)
    return result   

def plot_overall_best(grid, curves_output_path, dataset_name, pca = False):

    fig = plt.figure(figsize=(8, 4))

    gs0 = fig.add_gridspec(1, 1)
    gs0.update(wspace=0.0, hspace=0.2)
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec = gs0[0],
        wspace=0.05, width_ratios=[1, 0.03])

    # All axis we need
    contrast_ax = fig.add_subplot(gs1[0])
    colorbar_ax = fig.add_subplot(gs1[1])
    if pca == False:
        #grid = grid.map(fpf_2_gaussian_sigma)
        title = 'Overall Best Performance'
    else:
        title = 'Overall Best - PCA component'

    # Plot the contrast grid
    plot_contrast_grid(
        contrast_grid_axis=contrast_ax,
        colorbar_axis=colorbar_ax,
        contrast_grid=grid, 
        cmap = 'YlGn' if pca == True else "YlGnBu",
        vmax = np.max(grid) if pca == True else 2,
        )

    contrast_ax.set_ylabel(
        "Contrast - $c = f_p / f_*$ - [mag]", size=14)
    contrast_ax.set_xlabel(
        r"Separation [FWHM]", size=14)
    contrast_ax.set_title(
        "Contrast Grid: " + title,
        fontsize=16,
        fontweight="bold",
        y=1.03)

    contrast_ax.tick_params(
        axis='both',
        which='major',
        labelsize=12)

    # Save the figure
    fig.patch.set_facecolor('white')

    plt.savefig(f"{(curves_output_path)}/GRID_{title.replace(' ', '_')}_{dataset_name}.png", pad_inches=0.15)
    