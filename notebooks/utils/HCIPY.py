
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
import seaborn as sns
import pandas as pd

#images
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import torch
import imageio.v2 as imageio
from astropy.io import fits
from astropy.visualization import LogStretch, ImageNormalize, AsinhStretch
from astropy.modeling import models, fitting
from PIL import Image, ImageDraw, ImageFont
import matplotlib.gridspec as gridspec
import shutil 
from matplotlib.lines import Line2D
import glob
import tifffile as tiff
#import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter

#scientific libraries
from hcipy import *
import applefy
#importlib.reload(applefy)
from applefy import *
#importlib.reload(applefy.detections.contrast)
#from applefy.detections.contrast import Contrast
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.wrappers.pynpoint import MultiComponentPCAPynPoint
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, \
    fpf_2_gaussian_sigma, LaplaceBootstrapTest
from applefy.wrappers.vip import MultiComponentPCAvip

import fours
#importlib.reload(fours)
from fours.detection_limits.applefy_wrapper import CADIDataReductionGPU, PCADataReductionGPU, CADIDataReduction
from fours.models.rotation import FieldRotationModel

from skimage.registration import phase_cross_correlation


def chain(*elements):
    def wrapped(wf):
        for el in elements:
            wf = el(wf)
        return wf
    return wrapped


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

