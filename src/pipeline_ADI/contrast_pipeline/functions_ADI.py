
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
from scipy.ndimage import center_of_mass

#scientific libraries
from hcipy import *
import applefy
importlib.reload(applefy)
#importlib.reload(applefy.detections.contrast)
#from applefy.detections.contrast import Contrast
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, LaplaceBootstrapTest

import fours
importlib.reload(fours)
from fours.detection_limits.applefy_wrapper import CADIDataReductionGPU #, PCADataReductionGPU
from .pca_utils import PCADataReductionGPU
from applefy.detections.contrast import Contrast



import numpy as np
from skimage.registration import phase_cross_correlation


def estimate_center(stack, nframes=1000, upsample_factor=1000,
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
    if method == 'com':
        coords_maxpsf = (np.round(np.array(center_of_mass(img))).astype(int))
    if method == 'max':
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

    # num_parallel = cpu_count()//2

    if algo_name == 'PCAD':
        algorithm_function = PCADataReductionGPU(
            pca_numbers=fp_config['components'],
            device=fp_config['device'],
            pca_method=fp_config['pca_method'],
            niter=fp_config['niter'],
            random_state=fp_config['random_state'],
            eps=fp_config['eps'],
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

    if grid ==False:
        contrast_curves, contrast_errors = contrast_instance.compute_analytic_contrast_curves(
            statistical_test=statistical_test,
            confidence_level_fpf=gaussian_sigma_2_fpf(5),
            num_rot_iter=20,
            pixel_scale= pixel_scale)

        return contrast_curves, contrast_errors
    
    if grid == True:
        contrast_curves_grid, contrast_grids = contrast_instance.compute_analytic_contrast_grids(
            statistical_test=statistical_test,
            confidence_level_fpf=gaussian_sigma_2_fpf(5),
            num_rot_iter=20,
            safety_margin=1.0,
            num_cores=1, 
            pixel_scale= pixel_scale)

        return contrast_curves_grid, contrast_grids


def plot_contrast_grid(
    contrast_grid_axis,
    colorbar_axis,
    contrast_grid,
    cmap = "YlGnBu"):

    c_bar_kargs = dict(
        orientation = "vertical",
        label = r"Confidence [$\sigma_{\mathcal{N}}$]")

    heat = sns.heatmap(
        contrast_grid,
        vmax=2, vmin=7,
        annot=True,
        cmap= cmap,
        ax=contrast_grid_axis,
        cbar_ax=colorbar_axis,
        cbar_kws=c_bar_kargs)

    ylabels = ['{:.1f}'.format(float(x.get_text()))
               for x in heat.get_yticklabels()]
    _=heat.set_yticklabels(ylabels)
    xlabels = ['{:.1f}'.format(float(x.get_text()))
               for x in heat.get_xticklabels()]
    _=heat.set_xticklabels(xlabels)
    