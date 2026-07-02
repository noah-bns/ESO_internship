
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import os
from multiprocessing import cpu_count
from scipy import interpolate
import importlib
import shutil
from typing import Tuple, Callable, Optional
from typing import List, Dict, Union

#images
import torch
import imageio.v2 as imageio
from astropy.io import fits
from astropy.visualization import LogStretch, ImageNormalize, AsinhStretch
from astropy.modeling import models, fitting
from PIL import Image, ImageDraw

#scientific libraries
from hcipy import *
import applefy
importlib.reload(applefy)
#importlib.reload(applefy.detections.contrast)
#from applefy.detections.contrast import Contrast
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, \
    fpf_2_gaussian_sigma, LaplaceBootstrapTest

import fours
importlib.reload(fours)
from fours.detection_limits.applefy_wrapper import CADIDataReductionGPU #, PCADataReductionGPU
from .pca_utils import PCADataReductionGPU
from applefy.detections.contrast import Contrast


def zoom_to_peak(
        img, 
        radius):
    coords_maxpsf = (np.unravel_index(np.argmax(img, axis=None), img.shape))
    img = img[coords_maxpsf[0]-radius:coords_maxpsf[0]+radius+1, coords_maxpsf[1]-radius:coords_maxpsf[1]+radius+1]
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
    output_path: Path,
    dataset: dict,
    fp_config: dict,
    separations: np.ndarray,
    algo_name: str,
    angles: np.ndarray,
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

    # Remove existing directory and all its contents
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"Removed existing directory to avoid overwrites: {output_path}.")

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    flux_ratio = mag2flux_ratio(fp_config['flux_ratio_mag'])

    contrast_instance = Contrast(
        science_sequence=dataset["sci_img"],
        psf_template=dataset["psf"],
        parang_rad=angles,
        psf_fwhm_radius=dataset["fwhm"] / 2,
        dit_psf_template=dataset["dit_psf"],
        dit_science=dataset["dit_science"],
        scaling_factor=fp_config["scaling_factor"],
        checkpoint_dir= output_path
    )


    contrast_instance.design_fake_planet_experiments(
        flux_ratios= flux_ratio,
        num_planets=fp_config['num_fake_planets'],
        separations = separations,
        overwrite=True,
        )

    num_parallel = cpu_count()//2

    if algo_name == 'PCAD':
        algorithm_function = PCADataReductionGPU(
            pca_numbers=fp_config['components'],
            device=fp_config.get('device', 'auto'),
            pca_method=fp_config.get('pca_method', 'auto'),
            oversample=fp_config.get('oversample', 5),
            niter=fp_config.get('niter', 2),
            gram_threshold=fp_config.get('gram_threshold', 0.5),
            random_state=fp_config.get('random_state', None),
            eps=fp_config.get('eps', None),
            approx_svd_trunc=fp_config.get('approx_svd_trunc', None),
            subsample_rotation_grid=fp_config.get('subsample_rotation_grid', 1),
            combine=fp_config.get('combine', 'mean'),
        )

    if algo_name == 'CADI':
        algorithm_function = CADIDataReductionGPU(
            device = fp_config.get('device', 'auto')
                )
        
    try:
        contrast_instance.run_fake_planet_experiments(
            algorithm_function=algorithm_function,
            num_parallel=num_parallel)
    except:
        # can fail in multiprocessing, depending on whether optional dependencies are installed or not
        num_parallel=1
        contrast_instance.run_fake_planet_experiments(
            algorithm_function=algorithm_function,
            num_parallel=num_parallel)
                

    return contrast_instance


def compute_contrast_curves(
        contrast_instance, 
        fwhm, 
        pixel_scale, 
        photometry = 'FS', 
        test = 't-test'
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
    pixel_scale: pixel size in arcsec to convert FWHM
    
    Returns
    -------
    contrast_curves : object
        Computed analytic contrast curves.
    contrast_errors : object
        Uncertainties associated with the contrast curves.

    Warns
    -----
    UserWarning
        Raised if an unsupported photometry mode or statistical test is
        provided.
    """

    if photometry == 'FS':# Use spaced pixel values
        photometry_mode_planet = AperturePhotometryMode(
            "FS", # or "P"
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


    contrast_curves, contrast_errors = contrast_instance.compute_analytic_contrast_curves(
        statistical_test=statistical_test,
        confidence_level_fpf=gaussian_sigma_2_fpf(5),
        num_rot_iter=20,
        pixel_scale= pixel_scale)

    return contrast_curves, contrast_errors


def plot_contrast_curves(
        contrast_curves, 
        contrast_errors, 
        lim_mag_y, 
        lim_arcsec = None, 
        title = None,
        cmap = "magma",
        x_axis = None,
        x_axis_label = None
        ):
    # compute the overall best contrast curve
    PADI_values = contrast_curves.drop(columns=['cADI'], errors="ignore")
    overall_best = np.min(PADI_values.values, axis=1)

    # get the error bars of the the overall best contrast curve
    best_idx = np.argmin(PADI_values.values, axis=1)
    best_contrast_errors = contrast_errors.values[np.arange(len(best_idx)), best_idx]

    # Find one color for each number of PCA components used
    color_map = plt.cm.get_cmap(cmap)   # seaborn-style colormap available in mpl
    colors = [color_map(int(i)) for i in np.round(np.linspace(0, 220, len(PADI_values.columns)))]

    if x_axis:
        separations_arcsec = contrast_curves.reset_index(level=0).index * x_axis
    else:
        separations_arcsec = contrast_curves.reset_index(level=0).index
    separations_FWHM = contrast_curves.reset_index(level=1).index

    # 1.) Create Plot Layout
    fig = plt.figure(constrained_layout=False, figsize=(12, 8))
    gs0 = fig.add_gridspec(1, 1)
    axis_contrast_curvse = fig.add_subplot(gs0[0, 0])


    # ---------------------- Create the Plot --------------------
    i = 0 # color picker
    for tmp_model in contrast_curves.columns:
        
        if tmp_model == 'cADI':
            num_components = 'cADI'
            color = 'red'
        else:
            num_components = int(tmp_model[5:8])
            color = colors[i]
        tmp_flux_ratios = contrast_curves.reset_index(
            level=0)[tmp_model].values
        tmp_errors = contrast_errors.reset_index(
            level=0)[tmp_model].values

        axis_contrast_curvse.plot(
            separations_arcsec,
            tmp_flux_ratios,
            color = color,
            label=num_components)

        axis_contrast_curvse.fill_between(
            separations_arcsec,
            tmp_flux_ratios + tmp_errors,
            tmp_flux_ratios - tmp_errors,
            color = color,
            alpha=0.5)
        i+=1

    axis_contrast_curvse.set_yscale("log")
    # ------------ Plot the overall best -------------------------
    axis_contrast_curvse.plot(
        separations_arcsec,
        overall_best,
        color = "blue",
        lw=3,
        ls="--",
        label="Best")

    # ------------- Double axis and limits -----------------------
    if lim_arcsec:
        lim_arcsec_x = lim_arcsec
    else:
        lim_arcsec_x = (np.min(separations_arcsec) - 0.1, np.max(separations_arcsec) + 0.1)

    sep_lambda_arcse = interpolate.interp1d(
        separations_arcsec,
        separations_FWHM,
        fill_value='extrapolate')

    axis_contrast_curvse_mag = axis_contrast_curvse.twinx()
    axis_contrast_curvse_mag.plot(
        separations_arcsec,
        flux_ratio2mag(tmp_flux_ratios),
        alpha=0.)
    axis_contrast_curvse_mag.invert_yaxis()

    axis_contrast_curvse_lambda = axis_contrast_curvse.twiny()
    axis_contrast_curvse_lambda.plot(
        separations_FWHM,
        tmp_flux_ratios,
        alpha=0.)

    axis_contrast_curvse.grid(which='both')
    axis_contrast_curvse_mag.set_ylim(*lim_mag_y)
    axis_contrast_curvse.set_ylim(
        mag2flux_ratio(lim_mag_y[0]),
        mag2flux_ratio(lim_mag_y[1]))

    axis_contrast_curvse.set_xlim(
        *lim_arcsec_x)
    axis_contrast_curvse_mag.set_xlim(
        *lim_arcsec_x)
    axis_contrast_curvse_lambda.set_xlim(
        *sep_lambda_arcse(lim_arcsec_x))

    # ----------- Labels and fontsizes --------------------------
    if x_axis_label:
        x_axis_labelling = x_axis_label
    else:
        x_axis_labelling = r"Separation [arcsec]"
    axis_contrast_curvse.set_xlabel(
        x_axis_labelling, size=16)
    axis_contrast_curvse_lambda.set_xlabel(
        r"Separation [FWHM]", size=16)

    axis_contrast_curvse.set_ylabel(
        r"Planet-to-star flux ratio", size=16)
    axis_contrast_curvse_mag.set_ylabel(
        r"$\Delta$ Magnitude", size=16)

    axis_contrast_curvse.tick_params(
        axis='both', which='major', labelsize=14)
    axis_contrast_curvse_lambda.tick_params(
        axis='both', which='major', labelsize=14)
    axis_contrast_curvse_mag.tick_params(
        axis='both', which='major', labelsize=14)

    if title:
        set_title = title
    else:
        set_title = r"$5 \sigma_{\mathcal{N}}$ Contrast Curves"
    axis_contrast_curvse_mag.set_title(
        set_title,
        fontsize=18, fontweight="bold", y=1.1)

    # --------------------------- Legend -----------------------
    handles, labels = axis_contrast_curvse.\
        get_legend_handles_labels()

    leg1 = fig.legend(handles, labels,
                    bbox_to_anchor=(0.12, -0.1),
                    fontsize=14,
                    title="# PCA components",
                    loc='lower left', ncol=8)

    _=plt.setp(leg1.get_title(),fontsize=14)


def plot_best_PCA_number(
        contrast_curves_list, 
        components, 
        lables = None, 
        colors = ['red', 'blue']
        ):
    plt.figure(figsize=(12, 8))



    separations_arcsec = contrast_curves_list[-1].reset_index(level=0).index
    plt.plot(separations_arcsec,
            np.array(components)[np.argmin(
                contrast_curves_list[-1].values,
                axis=1)])
        
    plt.title(r"Best number of PCA components",
            fontsize=18, fontweight="bold", y=1.1)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Separation [arcsec]", fontsize=16)
    plt.ylabel("Number of PCA components", fontsize=16)
    plt.xlim(left = np.min(separations_arcsec), right = np.max(separations_arcsec))


    plt.grid()

    #create second x axis 
    #note: must have the same separations
    ax2 = plt.twiny()
    for i, contrast_curves in enumerate(contrast_curves_list):
        separations_FWHM = contrast_curves.reset_index(level=1).index

        ax2.plot(separations_FWHM,
                    np.array(components)[
                        np.argmin(contrast_curves.values, axis=1)],
                        label = lables[i] if lables else None,
                        color = colors[i]
                )

    ax2.set_xlabel("Separation [FWHM]", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_xlim(left = np.min(separations_FWHM), right = np.max(separations_FWHM))
    plt.legend(fontsize=14,
                    loc='upper right')


def create_pca_gif(
    folder,
    components,
    output_gif="pca_animation.gif",
    duration=0.5,
):
    """
    Create a GIF from residual FITS images corresponding to different PCA values.

    Parameters
    ----------
    folder : str or Path
        Base folder containing the 'residuals' directory.
    components : iterable
        PCA component numbers, e.g. [2, 5, 10, 100].
    residual_id : int
        Residual image ID (default: 0 -> residual_ID_0000.fits).
    output_gif : str
        Output GIF filename.
    duration : float
        Time per frame in seconds.
    """

    frames = []

    for pca in components:
        pca_str = f"{int(pca):03d}"

        fits_files = sorted(
            (
                Path(folder)
                / "residuals"
                / f"_PCA_{pca_str}_components"
            ).glob("*.fits")
        )

        for file in fits_files:

            data = fits.getdata(file)
            v = np.nanpercentile((data), 99.5)
            norm = ImageNormalize((data), vmin=-v, vmax=v, 
                                  #stretch=AsinhStretch(), 
                                  clip = True)
            frame = norm(data)
            frame = (250 * frame).astype(np.uint8)

            # Convert to RGB so we can draw colored text
            img = Image.fromarray(frame).convert("L") #instead of RGB
            draw = ImageDraw.Draw(img)
            label = (
                f"PCA = {int(pca)}\n"
                f"{file.name}"
            )

            draw.text(
                (10, 10),
                label,
                #fill=(0,0,0),
            )
            frames.append(np.array(img))

    imageio.mimsave(output_gif, frames, duration=duration)
    print(f"Saved GIF: {output_gif}")
