

import numpy as np
from pathlib import Path
import warnings
import os, glob
from multiprocessing import cpu_count
import importlib
import time
import gc
import pandas as pd
from typing import Optional, Tuple, Sequence


# scientific libraries
from sklearn.decomposition import PCA
from photutils.centroids import centroid_com
from photutils.aperture import aperture_photometry, CircularAperture
from scipy import interpolate
from scipy.ndimage import center_of_mass as nd_center_of_mass
from scipy.ndimage import gaussian_filter, shift, map_coordinates, fourier_shift
from skimage.registration import phase_cross_correlation
from applefy.utils.positions import center_subpixel
from numpy.fft import fftn, ifftn
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
from scipy import signal

# images
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams["hatch.linewidth"] = 0.5  # previous pdf hatch linewidth
import tifffile as tiff
from matplotlib.patches import Circle
from photutils.aperture import CircularAperture


# import pre_process.statistics as stat
# importlib.reload(stat)
from pre_process.statistics import *
# import utils.plot_pre_processing as plot
# importlib.reload(plot)
# from utils.plot_pre_processing import *





def centroid_com(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Simple center-of-mass centroid. Returns (yc, xc) in image coordinates.
    """
    if mask is None:
        return nd_center_of_mass(image)
    # apply mask
    arr = np.array(image, dtype=float)
    arr = arr * (mask.astype(float))
    if arr.sum() <= 0:
        # fallback to geometric center if nothing left
        ny, nx = image.shape
        return ( (ny - 1) / 2.0, (nx - 1) / 2.0 )
    return nd_center_of_mass(arr)


def _rotated_gaussian(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """Rotated 2D Gaussian used by curve_fit (coords is (2, N) meshgrid)."""
    (x, y) = coords
    xo = float(x0)
    yo = float(y0)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    g = offset + amplitude * np.exp( - (a * ((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c * ((y-yo)**2)))
    return g.ravel()

def fit_2d_gaussian(image: np.ndarray, bbox: Optional[Tuple[int,int,int,int]] = None,
                    p0: Optional[Sequence[float]] = None, maxfev: int = 2000) -> Tuple[float,float, dict]:
    """
    Fit a rotated 2D Gaussian to `image` (or a bbox). Returns (yc, xc, popt_dict).

    bbox: optional (y0, y1, x0, x1) to restrict fit region.
    p0: initial guess [amp, x0, y0, sx, sy, theta, offset] — x0, y0 are in bbox coordinates.
    """
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        im = image[y0:y1, x0:x1]
    else:
        im = image

    ny, nx = im.shape
    Y, X = np.mgrid[0:ny, 0:nx]

    # initial guesses: amplitude, x0, y0, sx, sy, theta, offset
    if p0 is None:
        amp0 = float(im.max() - np.median(im))
        com_y, com_x = centroid_com(im)
        sx0 = sy0 = max(1.0, min(ny, nx) / 8.0)
        p0 = [amp0, com_x, com_y, sx0, sy0, 0.0, float(np.median(im))]

    try:
        popt, pcov = curve_fit(_rotated_gaussian, (X, Y), im.ravel(), p0=p0, maxfev=maxfev)
    except Exception:
        # fallback to center-of-mass if fit fails
        com_y, com_x = centroid_com(im)
        return (float(com_y + (y0 if bbox is not None else 0)),
                float(com_x + (x0 if bbox is not None else 0)),
                {"method":"com_fallback"})

    amp, x0f, y0f, sx, sy, theta, offset = popt
    yc = float(y0f + (y0 if bbox is not None else 0))
    xc = float(x0f + (x0 if bbox is not None else 0))
    return yc, xc, {"amplitude":amp, "sigma_x":sx, "sigma_y":sy, "theta":theta, "offset":offset}

# -----------------------
# (C) Coronagraphic mask detection & circle fit
# -----------------------
def _algebraic_circle_fit(xs: np.ndarray, ys: np.ndarray) -> Tuple[float,float,float]:
    """
    Algebraic least-squares circle fit.
    Returns (yc, xc, r)
    """
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    a1, a2, c0 = c
    xc = 0.5 * a1
    yc = 0.5 * a2
    r = np.sqrt(xc**2 + yc**2 + c0)
    return yc, xc, r

def detect_coron_mask_and_fit_circle(image: np.ndarray, smooth_sigma_annulus: float = 10.0,
                                     threshold_frac: float = 0.7, max_area: int = 100) -> Tuple[float,float,float,np.ndarray]:
    """
    Detect a dark circular occulting mask (coronagraph) and fit its center.

    Returns (yc, xc, radius, mask) where mask is boolean for the occulting area.
    """
    im = np.asarray(image, dtype=float)

    # smooth to suppress speckles
    im_s = gaussian_filter(im, smooth_sigma_annulus)

    # threshold low-intensity region
    thr = np.max(im_s) * threshold_frac
    bw = im_s > thr

    # keep largest connected component (assumed to be the occulting mask)
    labeled, nl = ndi.label(bw)
    if nl == 0:
        raise RuntimeError("No dark region found for coronagraph mask detection.")
    sizes = ndi.sum(bw, labeled, range(1, nl+1))
    if sizes.size == 0:
        raise RuntimeError("No components found.")
    largest_label = np.argmax(sizes) + 1
    mask = (labeled == largest_label)
    # if mask.sum() > max_area:
    #     raise RuntimeError("Detected mask too large / noisy detection.")

    # boundary points
    eroded = ndi.binary_erosion(mask, iterations=2)
    boundary = mask ^ eroded
    ys, xs = np.nonzero(boundary)
    if len(xs) < 10:
        # fallback to full mask points
        ys, xs = np.nonzero(mask)

    # fit circle
    yc, xc, r = _algebraic_circle_fit(xs.astype(float), ys.astype(float))
    return yc, xc, r, mask



def estimate_center_correlation(
    stack,
    nframes,
    upsample_factor=100,
    max_iter=100,
    tol=1e-2,
    ref_psf=None,
    search_radius=7,
):

    """
    Estimate the center of a stack using iterative 90° rotational
    phase cross-correlation.

    Conceptual, not very robust. Use with caution.

    Parameters
    ----------
    stack : ndarray
        Image stack whose center is to be estimated.

    ref_psf : smoothed uncorrelated PSFs. The median of this stack is used to
        estimate the approximate PSF peak, which constrains the search.

    search_radius : int, optional
        Radius (pixels) of the allowed search region around the reference
        peak. If None, no constraint is applied.

    smooth_sigma : float
        Gaussian smoothing before locating the brightest pixel.

    Returns
    -------
    yc, xc : float
        Estimated subpixel center.
    """

    ref_frame = np.median(stack[:nframes], axis=0)
    ny, nx = ref_frame.shape

    # ------------------------------------------------------------------
    # Initial center
    # ------------------------------------------------------------------
    if ref_psf is not None:
        
        yc0, xc0 = np.unravel_index(np.argmax(ref_psf), ref_psf.shape)

        yc = float(yc0)
        xc = float(xc0)
    else:
        yc = (ny - 1) / 2
        xc = (nx - 1) / 2
        yc0, xc0 = yc, xc

    # ------------------------------------------------------------------
    # Iterative refinement
    # ------------------------------------------------------------------
    for _ in range(max_iter):

        half = int(min(yc, xc, ny - 1 - yc, nx - 1 - xc))

        y = int(round(yc))
        x = int(round(xc))

        sub = ref_frame[y - half : y + half + 1, x - half : x + half + 1]

        rot = np.rot90(sub, 1)

        shift, _, _ = phase_cross_correlation(sub, rot, upsample_factor=upsample_factor)

        dy_corr, dx_corr = shift 

        dx = (dx_corr - dy_corr) / 2
        dy = (dx_corr + dy_corr) / 2

        yc += dy
        xc += dx

        # Constrain search around reference PSF center
        if ref_psf is not None:
            yc = np.clip(yc, yc0 - search_radius, yc0 + search_radius)
            xc = np.clip(xc, xc0 - search_radius, xc0 + search_radius)


        if np.hypot(dy, dx) < tol:
            print(f"Center converged after {_+1} iterations.")
            break

    return (yc, xc)