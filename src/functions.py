
from fours.utils import pca
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
import os
from multiprocessing import cpu_count
from scipy import interpolate
import importlib
import time
import gc

#images
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import torch
import imageio.v2 as imageio
from astropy.io import fits
from astropy.visualization import LogStretch, ImageNormalize, AsinhStretch
from astropy.modeling import models, fitting
from PIL import Image, ImageDraw, ImageFont

#scientific libraries
from hcipy import *
import applefy
importlib.reload(applefy)
from applefy import *
#importlib.reload(applefy.detections.contrast)
#from applefy.detections.contrast import Contrast
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.wrappers.pynpoint import MultiComponentPCAPynPoint
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, \
    fpf_2_gaussian_sigma, LaplaceBootstrapTest

import fours
importlib.reload(fours)
from fours.detection_limits.applefy_wrapper import CADIDataReductionGPU, PCADataReductionGPU, CADIDataReduction
from fours.models.rotation import FieldRotationModel



def optimal_svht_coef(beta):
    """beta = m/n where m >= n."""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def gavish_donoho_rank(img, sigma=None):
    """
    S: singular values, m x n matrix shape.
    sigma: noise std; if None, estimate from median singular value.
    """
    _, S, _ = torch.linalg.svd(img)
    S = np.asarray(S)

    m,n = img.shape
    beta = min(m, n) / max(m, n)
    if sigma is None:
        # Estimate sigma from median singular value
        sigma = np.median(S) / (np.sqrt(2) * 0.6745)
    tau = optimal_svht_coef(beta) * np.sqrt(max(m, n)) * sigma
    mask = (S > tau)
    return int(mask.sum())



def zoom_to_peak(
        img, 
        radius):
    coords_maxpsf = (np.unravel_index(np.argmax(img, axis=None), img.shape))
    img = img[coords_maxpsf[0]-radius:coords_maxpsf[0]+radius+1, coords_maxpsf[1]-radius:coords_maxpsf[1]+radius+1]
    return img


def chain(*elements):
    def wrapped(wf):
        for el in elements:
            wf = el(wf)
        return wf
    return wrapped


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
    

def generate_focal_plane(
        wavelength_sci, 
        pupil_diameter, 
        num_pupil_pixels,
        pupil_grid_diameter, 
        q, 
        num_airy, 
        telescope = make_vlt_aperture()):
    """
    Generate the telescope pupil and focal-plane propagator.

    Parameters
    ----------
    wavelength_sci : float
        Science wavelength [m].
    pupil_diameter : float
        Telescope diameter [m].
    num_pupil_pixels : int
        Number of pixels across the pupil grid.
    pupil_grid_diameter : float
        Physical size of the pupil grid [m].
    q : float
        Focal-plane oversampling factor.
    num_airy : float
        Size of the focal plane in Airy radii.
    telescope:
        give the aperture generator function, by default: generates a VLT aperture.

    Returns
    -------
    telescope_pupil : Field
        Telescope aperture mask.
    pupil_grid : Grid
        HCIPy pupil grid.
    propagator : FraunhoferPropagator
        Propagator from pupil plane to focal plane.
    """
    spatial_resolution = wavelength_sci / pupil_diameter
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
    telescope_pupil = telescope(pupil_grid)
    focal_grid = make_focal_grid(
        q=q,
        num_airy=num_airy,
        spatial_resolution=spatial_resolution
    )
    propagator = FraunhoferPropagator(pupil_grid, focal_grid)

    return telescope_pupil, pupil_grid, propagator


def vlt_lyot_stop(
        grid, 
        Lyot_ratio = 1.05, 
        spider_ratio = 3, 
        outer_ratio = None):

    if not outer_ratio:
        outer_ratio = Lyot_ratio
    #geometry
    spider_width            = 0.040   * Lyot_ratio * spider_ratio   # meter
    spider_offset           = 0.4045                                # meter
    spider_outer_radius     = 4.2197                                # meter
    angle_between_spiders   = 101                                   # degrees
    pupil_diameter          = 8.0     * (2 - outer_ratio)            # meter
    central_obscuration_ratio = 1.116 / pupil_diameter * Lyot_ratio

    obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)

    spider_inner_radius = spider_offset / np.cos(np.radians(45 - (angle_between_spiders - 90) / 2))
    
    spider_start = spider_inner_radius * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    spider_end_1 = spider_outer_radius * np.array([np.cos(np.pi), np.sin(np.pi)])
    spider_end_2 = spider_outer_radius * np.array([np.cos(-np.pi / 2), np.sin(-np.pi / 2)])
    spider_end_3 = spider_outer_radius * np.array([np.cos(0), np.sin(0)])
    spider_end_4 = spider_outer_radius * np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])

    #generate the spiders
    spider1 = make_spider(-spider_start, spider_end_1, spider_width)
    spider2 = make_spider(-spider_start, spider_end_2, spider_width)
    spider3 = make_spider( spider_start, spider_end_3, spider_width)
    spider4 = make_spider( spider_start, spider_end_4, spider_width)

    return Field(obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid), grid)


def create_wavefront(
        telescope_pupil, 
        pupil_grid, 
        wavelength_sci, 
                     stellar_magnitude = None, 
                     zero_magnitude_flux = 1.7e10, #photon/s VLT value (from Jalo)
                     frame_rate = 1, #s
                     angular_separation = 0, 
                     position_angle = 0,   
                     num_photons_star = 1, 
                     contrast = 1):
    """
    The function creates wavefront objects, with the possibility to assign it an intensity (in terms of absolute magnitude, or directly flux), and a phase.
    
    Inputs:
    ----------
    telescope_pupil:        pupil object (field)
    pupil_grid:             pupil grid
    wavelength_sci:         wavefront wavelength in meters
    stellar_magnitude:      optional absolute magnitude of the star in field of view
    zero_magnitude_flux:    float in photons per second, instrument and telescope and filter dependent, default value = 1.7e10, #photon/s VLT value (from Jalo)
    frame_rate = 1:         integration time of the frame in seconds, i.e. AO loop duration if the WF is used to create phase frames
    angular_separation:     if off centred object, distance in lambda/D, default = centred object
    position_angle :        float Position angle in radians. Measured counterclockwise from +x.
    num_photons_star:       flux in photons per second of the star in field of view, by default correspond to a normalised intensity of 1
    contrast:               float correspondind to the ratio of WF intensity over the intensity of the brightest star in field of view, dimensionless, by default zero
    
    Return:
    -----------
    wf:                     wavefront object with an integrated intensity for a "frame_rate" duration.
    """

    if stellar_magnitude:
        num_photons_star = zero_magnitude_flux * 10**(-stellar_magnitude/2.5)
    # Phase ramp direction
    x_rot = (
        pupil_grid.x * np.cos(position_angle)
        + pupil_grid.y * np.sin(position_angle)
    )

    pos = telescope_pupil * np.exp(
        2j * np.pi * x_rot * angular_separation
    )

    wf = Wavefront(pos, wavelength_sci)
    wf.total_power = num_photons_star * frame_rate * contrast
    return wf


def phase2apodizer(
        phase_screen, 
        telescope_pupil, 
        pupil_grid = None):
    if not isinstance(phase_screen, Field):
        phase_screen = Field(phase_screen, pupil_grid).flatten()
    phase_screen[telescope_pupil.astype(bool)] -= np.mean(
        phase_screen[telescope_pupil.astype(bool)]
    )
    phase = SurfaceApodizer(
        .5 * phase_screen, #bcs the -1 refractive index doubles the aberration
        refractive_index=-1
    )
    return phase


def generate_aberrated_frames(
        telescope_pupil, 
        propagator,
        pupil_grid_diameter, 
        pupil_grid,
        wavelength_sci, 
        wavefronts,
        additional_phase = None,
        ptv=None, 
        coro=None,
        fried_parameter=None, 
        outer_scale=None,
        velocity=None, 
        shot_noise=False):
    """
    Simulate a coronagraphic science image with optional aberrations,
    atmospheric turbulence, and photon noise.

    Parameters
    ----------
    telescope_pupil : HCIPy object
        Telescope aperture.
    propagator : HCIPy function
        Propagation function of the wavefront.
    pupil_grid_diameter : float
        Physical size of the pupil grid [m].
    pupil_grid : HCIPy grid
        Pupil grid.
    wavelength_sci : float
        Science wavelength [m].
    wavefronts : list, object
        list of wavefronts to propagate
    additional_phase : Field, optional,
        Additional phase screen to add.
    ptv : float, optional
        Peak-to-valley non-common path aberration amplitude.
    coro : int, optional
        Coronagraph propagation function.
    fried_parameter : float, optional
        Fried parameter r0 for atmospheric turbulence.
    outer_scale : float, optional
        Atmospheric outer scale.
    velocity : float or array-like, optional
        Wind velocity for turbulence simulation.
    shot_noise : bool, optional
        If True, add Poisson shot noise.

    Returns
    -------
    science_img : ndarray or Field
        Simulated focal-plane intensity image.
    """

    if not isinstance(wavefronts, list):
        wavefronts = [wavefronts]

    if fried_parameter:
        # atmosphere
        Cn_squared = Cn_squared_from_fried_parameter(
            fried_parameter,
            wavelength_sci
        )
        layer = InfiniteAtmosphericLayer(
            pupil_grid,
            Cn_squared,
            outer_scale,
            velocity)
        wavefronts = [layer(wf) for wf in wavefronts]

    if additional_phase is not None:
        AO_phase = phase2apodizer(additional_phase, telescope_pupil)
        wavefronts = [AO_phase(wf) for wf in wavefronts]

    if coro:
        wavefronts = [coro(wf) for wf in wavefronts]

    if ptv:
        # ncpa
        ncpas = make_power_law_error(
            pupil_grid,
            ptv,
            pupil_grid_diameter,
            -2.5
        ).squeeze()
        ncpa_surface = phase2apodizer(ncpas, telescope_pupil)
        wavefronts = [ncpa_surface(wf) for wf in wavefronts]

    science_img = propagator.forward(wavefronts[0]).power * 0
    for wf in wavefronts:
        science_img += propagator.forward(wf).power

    if shot_noise == True:
        # Simulate photon noise
        science_img = large_poisson(science_img)

    return science_img


def fake_planet_experiment(
        contrast_instance, 
        flux_ratio, 
        num_fake_planets, 
        components,
        version = 'PCAD',
        separations = None,
        approx_svd = -1,
        device = 'cpu'
        ):
    """
    Run a fake planet injection and recovery experiment following the
    Appleby tutorial workflow.
    see: https://applefy.readthedocs.io/en/latest/02_user_documentation/01_contrast_curves.html

    The function first generates synthetic companions with the specified
    flux ratio, then processes the data using a multi-component PCA-based
    PSF subtraction algorithm (with PynPoint). Parallel execution is attempted by default;
    if multiprocessing fails (e.g., due to optional dependency issues),
    the experiment is rerun in serial mode.

    Parameters
    ----------
    contrast_instance : object
        Contrast analysis instance containing the dataset and experiment
        configuration.
    flux_ratio : float or array-like
        Flux ratio(s) used for the injected fake planets.
    num_fake_planets : int
        Number of fake planets to inject.
    components : int or list[int]
        Number of PCA components used in the
        ``MultiComponentPCAPynPoint`` algorithm.
    """

    contrast_instance.design_fake_planet_experiments(
        flux_ratios=flux_ratio,
        num_planets=num_fake_planets,
        separations = separations,
        overwrite=True,
        )

    num_parallel = cpu_count()//2

    #algorithm_function = MultiComponentPCAPynPoint(
     #   num_pcas=components,
      #  scratch_dir=contrast_instance.scratch_dir,
       # num_cpus_pynpoint=1)
    if version == 'PCAD':
        algorithm_function = PCADataReductionGPU(
                pca_numbers = components,
                approx_svd = approx_svd, # truncated for large tensor calculations 
                device = device
                )
    if version == 'CADI':
        algorithm_function = CADIDataReductionGPU(
            device = device
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
        x_axis_label = None,
        alpha = 0.7,
        ):
    # compute the overall best contrast curve
    PADI_values = contrast_curves.loc[:, ~contrast_curves.columns.str.contains("cADI", regex=True)]
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
        
        if 'cADI'.lower() in tmp_model.lower():
            num_components = 'cADI'
            color = 'red'
        else:
            num_components = int(tmp_model[5:8])
            color = colors[i]
            i+=1
        tmp_flux_ratios = contrast_curves.reset_index(
            level=0)[tmp_model].values
        tmp_errors = contrast_errors.reset_index(
            level=0)[tmp_model].values

        axis_contrast_curvse.plot(
            separations_arcsec,
            tmp_flux_ratios,
            color = color,
            alpha = alpha,
            label=num_components)

        axis_contrast_curvse.fill_between(
            separations_arcsec,
            tmp_flux_ratios + tmp_errors,
            tmp_flux_ratios - tmp_errors,
            color = color,
            alpha=alpha/2)
        

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


def comparison(curves1, errs1, curves2, errs2,
               title=None,
               cmap="winter"):


    delta = curves1 - curves2

    fig, ax = plt.subplots(figsize=(12, 8))

    color_map = plt.cm.get_cmap(cmap)
    colors = [color_map(i) for i in np.linspace(0, 1, len(delta.columns))]

    for i, col in enumerate(delta.columns):

        # align everything first
        c1 = curves1[col]
        c2 = curves2[col]

        common_index = c1.index.intersection(c2.index)

        c1 = c1.loc[common_index]
        c2 = c2.loc[common_index]

        e1 = errs1[col].loc[common_index]
        e2 = errs2[col].loc[common_index]

        # NOW everything has SAME length
        y = (c1 - c2).values
        x_local = common_index.get_level_values(0).to_numpy()

        overlap = np.abs(c1.values - c2.values) <= (e1.values + e2.values)

        ax.fill_between(
            x_local,
            (e1.values + e2.values),
            - (e1.values + e2.values),
            facecolor='none',
            edgecolor='black' if col == 'cADI' else colors[i],
            hatch='/',
            linewidth=0.2,
            label = 'Cumulative error bar' if i==len(delta.columns)-1 else None
        )

        ax.plot(
            x_local,
            y,
            color='black' if col =='cADI' else colors[i],
            label='CADI' if col =='cADI' else f'{int(col[5:8])} PCA',
            alpha = 0.7
        )

        ax.scatter(
            x_local[overlap],
            y[overlap],
            color="green",
            s=25,
            zorder=11
        )

        ax.scatter(
            x_local[~overlap],
            y[~overlap],
            color="red",
            s=25,
            zorder=11
        )

        
    ax.axhline(0, color="black", ls="--", linewidth=0.4)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_ylabel("Planet-to-Star flux ratio")
    ax.set_xlabel("Separation (FWHM)")
    ax.grid(True)

    if title:
        ax.set_title(title)
    
    component_handles, component_labels = ax.get_legend_handles_labels()

    status_handles = [
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor='green', markersize=8,
            label='Error bars overlap'),
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor='red', markersize=8,
            label='No overlap')
    ]

    ax.legend(
        component_handles + status_handles,
        component_labels + ['Error bars overlap', 'No overlap'],
        fontsize=10,
        loc = 'upper right',
        ncols = 2
    )

    return fig, ax


def _get_pca_fitter(
    pca_method: str,
    n_components: int,
    n_samples: int,
    n_features: int,
    gram_threshold: float = 0.5,
    oversample: int = 5,
    niter: int = 2,
    eps: float | None = None,
) -> callable:
    """
    Select and return the appropriate PCA fitting function based on method and data shape.
    
    Returns a callable that takes (X) and returns (components, singular_values).
    """
    
    # Auto-select method if needed
    if pca_method == "auto":
        if n_components >= 0.8 * min(n_samples, n_features):
            method = "svd"
        elif n_samples <= gram_threshold * n_features:
            method = "gram"
        else:
            method = "lowrank"
    else:
        method = pca_method
    
    if method == "svd":
        def fitter(X: torch.Tensor) -> torch.Tensor:
            """Exact reduced SVD."""
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
            components = Vh[:n_components]
            #singular_values = S[:n_components]
            return components #, singular_values
        
        return fitter, method
    
    elif method == "gram":
        def fitter(X: torch.Tensor) -> torch.Tensor:
            """Exact PCA via sample Gram matrix."""
            eps_val = eps if eps is not None else float(torch.finfo(X.dtype).eps)
            
            C = X @ X.T  # (n_samples, n_samples)
            evals, U = torch.linalg.eigh(C)
            
            # Sort by largest eigenvalues
            idx = torch.argsort(evals, descending=True)
            evals = evals[idx]
            U = U[:, idx]
            
            evals = evals[:n_components]
            U = U[:, :n_components]
            
            singular_values = torch.sqrt(torch.clamp(evals, min=0.0))
            
            # Filter out invalid singular values
            valid = singular_values > eps_val
            if not torch.any(valid):
                raise RuntimeError("All singular values are numerically zero.")
            
            U_valid = U[:, valid]
            S_valid = singular_values[valid]
            
            # Compute right singular vectors: V = X.T @ U @ diag(1/S)
            V = X.T @ (U_valid / S_valid)
            
            # Improve numerical orthogonality
            V, _ = torch.linalg.qr(V, mode="reduced")
            
            components = V[:, :n_components].T
            
            # Pad if needed to maintain consistent shapes
            if components.shape[0] < n_components:
                missing = n_components - components.shape[0]
                pad_components = torch.zeros(
                    missing,
                    X.shape[1],
                    device=X.device,
                    dtype=X.dtype,
                )
                components = torch.cat([components, pad_components], dim=0)
                
            #     pad_singular_values = torch.zeros(
            #         missing,
            #         device=X.device,
            #         dtype=X.dtype,
            #     )
            #     singular_values = torch.cat([S_valid, pad_singular_values], dim=0)
            # else:
            #     singular_values = singular_values[:n_components]
            
            return components #, singular_values
        
        return fitter, method
    
    elif method == "lowrank":
        def fitter(X: torch.Tensor) -> torch.Tensor:
            """Randomized approximate low-rank PCA."""
            q = min(
                n_components + oversample,
                min(X.shape),
            )
            
            _, S, V = torch.pca_lowrank(
                X,
                q=q,
                center=False,
                niter=niter,
            )
            
            components = V[:, :n_components].T
            #singular_values = S[:n_components]
            return components #, singular_values
        
        return fitter, method
    
    else:
        raise ValueError(f"Invalid PCA method: {pca_method}")
    


def pca_psf_subtraction_gpu(
        images: np.ndarray,
        angles: np.ndarray,
        pca_numbers: np.ndarray,
        device: str = "auto",
        pca_method: str = "auto",
        oversample: int = 5,
        niter: int = 2,
        gram_threshold: float = 0.5,
        random_state: int | None = None,
        eps: float | None = None,
        subsample_rotation_grid: int = 1,
        verbose: bool = False,
        combine: str = "mean",
        dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """
    PCA-based PSF subtraction using PCATorch.
    
    Parameters
    ----------
    images : np.ndarray
        Image cube with shape (n_frames, height, width)
    angles : np.ndarray
        Parallactic angles with shape (n_frames,)
    pca_numbers : np.ndarray
        PCA component numbers to evaluate
    pca_method : str
        PCA method: "auto", "svd", "gram", or "lowrank"
    """

    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else "cpu")

    pca_numbers = np.asarray(pca_numbers, dtype=int)

    if pca_numbers.ndim != 1:
        raise ValueError(f"Expected pca_numbers to be 1D, got {pca_numbers.shape}.")

    if np.any(pca_numbers < 1):
        raise ValueError("All PCA numbers should be >= 1.")
    

    with torch.no_grad():
        t0 = time.perf_counter()

        # 1.) Convert images to torch tensor
        im_shape = images.shape
        #images_torch = torch.from_numpy(images).to(device)
        n_frames, height, width = im_shape
        images_torch = torch.as_tensor(images, device=device, dtype=dtype)

        t1 = time.perf_counter()
        print(f"[Timing] Convert to tensor: {t1 - t0:.6f}s")

        # 2.) remove the mean as needed for PCA
        images_torch = images_torch - images_torch.mean(dim=0)

        # 3.) reshape images to fit for PCA
        #images_flat = images_torch.view(im_shape[0], im_shape[1] * im_shape[2])
        images_flat = images_torch.reshape(n_frames, height * width)
        
        # 4.) Fit PCA using PCATorch
        if verbose:
            print(f"Fit PCA ({pca_method}) ...", end="")

        
        #### 
        # based off https://github.com/markusbonse/near_processing/blob/main/near_processing/utils/pca.py#L11 PCATorch class
        ####

        # pca = PCATorch(
        #     n_components=max_components,
        #     method=pca_method,
        #     center=True,  # Remove mean
        #     device=device,
        #     dtype=images_torch.dtype,
        #     oversample=oversample,
        #     niter=niter,
        #     gram_threshold=gram_threshold,
        #     random_state=random_state,
        #     eps=eps,
        # )
        
        # pca.fit(images_flat)

        # Get the appropriate PCA fitter

        max_components = int(np.max(pca_numbers))
        n_samples, n_features = images_flat.shape
        max_rank = min(n_samples, n_features)

        if max_components > max_rank:
            raise ValueError(
                f"n_components={max_components} is larger than "
                f"min(n_samples, n_features)={max_rank}."
            )

        if random_state is not None:
            torch.manual_seed(random_state)

        pca_fitter, method_used = _get_pca_fitter(
            pca_method=pca_method,
            n_components=max_components,
            n_samples=n_frames,
            n_features=height * width,
            gram_threshold=gram_threshold,
            oversample=oversample,
            niter=niter,
            eps=eps,
        )

        # Fit PCA once
        components = pca_fitter(images_flat)

        if verbose:
            print(f"[DONE] (method: {method_used})")

        t2 = time.perf_counter()
        print(f"[Timing] Compute PCA Basis: {t2 - t1:.6f}s")

        # 5.) Build rotation model
        rotation_model = FieldRotationModel(
            all_angles=angles,
            input_size=im_shape[1],
            subsample=subsample_rotation_grid,
            inverse=False,
            register_grid=True
        ).to(device)

        t3 = time.perf_counter()
        print(f"[Timing] Field rotation: {t3 - t2:.6f}s")

        # 6.) Compute PCA residuals for all given PCA numbers
        pca_residuals = []
        if verbose:
            print("Compute PCA residuals ...", end="")

        for pca_number in pca_numbers:
            pca_number = int(pca_number)


            # Project onto PCA components
            pca_scores = images_flat @ components.T  # shape: (n_frames, pca_number)
            
            # Reconstruct noise model
            noise_estimate = pca_scores @ components  # shape: (n_frames, n_features)
            
            # Compute residuals
            residual = images_flat - noise_estimate
            residual_sequence = residual.view(im_shape[0], im_shape[1], im_shape[2])
            del residual, noise_estimate, pca_scores

            # Subtract temporal median if not using mean combine
            if combine != "mean":
                residual_sequence = residual_sequence - torch.median(residual_sequence, dim=0)[0]

            # Derotate frames
            rotated_frames = rotation_model(
                residual_sequence.unsqueeze(1).float(),
                parang_idx=torch.arange(len(residual_sequence), device=device)
            ).squeeze(1)
            del residual_sequence

            # Combine derotated frames
            if combine == "mean":
                residual_final = torch.mean(rotated_frames, dim=0).cpu().numpy()
            elif combine == "median":
                residual_final = torch.median(rotated_frames, dim=0)[0].cpu().numpy()
            else:
                raise ValueError(f"Invalid combine method: {combine}")
            
            pca_residuals.append(residual_final)
            del rotated_frames, residual_final

            if verbose:
                print("[DONE]")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        t4 = time.perf_counter()
        print(f"[Timing] PCA Residuals: {t4 - t3:.6f}s")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.3f}GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.3f}GB")

        return np.array(pca_residuals)