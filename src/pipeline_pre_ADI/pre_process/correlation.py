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
import torch

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
from astropy.modeling import models, fitting

# images
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams["hatch.linewidth"] = 0.5  # previous pdf hatch linewidth
import tifffile as tiff
from matplotlib.patches import Circle
from photutils.aperture import CircularAperture


from utils.statistics import *
from utils.naming import *


def symmetric_intensities(
    cube,
    angles_deg,
    r_min,
    r_max,
    dr=5,
    photometry_radius=0.5,
):
    # ============================================================
    # Symmetric intensity function
    # ============================================================

    """
    Extract intensities along a radial line and its 180-degree
    symmetric counterpart.

    Parameters
    ----------
    cube : ndarray
        Image cube with shape (n_frames, ny, nx).

    angle_deg : float or array-like
        Angle of the radial line in degrees.
        0 deg = +x direction
        90 deg = +y direction

    r_min : float
        Minimum radial distance from the center.

    r_max : float
        Maximum radial distance from the center.

    dr : float
        Radial separation between samples in pixels.

    Returns
    -------

    left : ndarray
        Intensities along the selected radial direction.
        Shape = (n_angles, n_frames, n_radii).

    symmetric : ndarray
        Intensities at the 180-degree symmetric positions.
        Shape = (n_angles, n_frames, n_radii).

    """

    n_frames, ny, nx = cube.shape

    # Make sure even a single angle becomes an array
    angles_deg = np.atleast_1d(angles_deg)
    angles_rad = np.deg2rad(angles_deg)

    # --------------------------------------------------------
    # Sub-pixel center
    # --------------------------------------------------------
    cy, cx = center_subpixel(cube[0])

    # --------------------------------------------------------
    # Radial coordinates
    # --------------------------------------------------------
    radii = np.arange(
        r_min + 0.5 * dr + photometry_radius,
        r_max - photometry_radius,
        (dr + 2 * photometry_radius),
    )

    n_angles = len(angles_deg)
    n_radii = len(radii)

    # --------------------------------------------------------
    # Coordinates along selected direction
    # --------------------------------------------------------
    x1 = (cx + radii[None, :] * np.cos(angles_rad)[:, None]).T.flatten()
    y1 = (cy + radii[None, :] * np.sin(angles_rad)[:, None]).T.flatten()

    # --------------------------------------------------------
    # 180-degree symmetric coordinates
    # --------------------------------------------------------
    x2 = (cx - radii[None, :] * np.cos(angles_rad)[:, None]).T.flatten()
    y2 = (cy - radii[None, :] * np.sin(angles_rad)[:, None]).T.flatten()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------
    left = np.empty((n_frames, n_radii * n_angles), dtype=cube.dtype)

    symmetric = np.empty_like(left)

    # --------------------------------------------------------
    # Extract intensities
    # --------------------------------------------------------
    for i, frame in enumerate(cube):

        tmp_apertures_left = CircularAperture(
            positions=np.column_stack((x1, y1)), r=photometry_radius
        )

        left[i, :] = aperture_photometry(frame, tmp_apertures_left, method="center")[
            "aperture_sum"
        ]

        # left[i, :] = map_coordinates(
        #     cube[i],
        #     [y1, x1],
        #     order=1,
        #     mode="nearest"
        # )

        # symmetric[i, :] = map_coordinates(
        #     cube[i],
        #     [y2, x2],
        #     order=1,
        #     mode="nearest"
        # )
        tmp_apertures_sym = CircularAperture(
            positions=np.column_stack((x2, y2)), r=photometry_radius
        )

        symmetric[i, :] = aperture_photometry(
            frame, tmp_apertures_sym, method="center"
        )["aperture_sum"]

    return (x1, y1), left, (x2, y2), symmetric, radii


def symmetric_analysis_method(
    data_cube,
    name,
    extraname="",
    angle=45.0,
    inner_rad=8,
    outer_rad=32,
    dr=5,
    crop=20,
    photometry_radius=0.5,
):

    data_cube_cropped = data_cube[:, crop:-crop, crop:-crop]

    _, left_int, _, symmetric_int, radii = symmetric_intensities(
        data_cube_cropped,
        angle,
        inner_rad,
        outer_rad,
        dr=dr,
        photometry_radius=photometry_radius,
    )

    results = []

    if isinstance(angle, list):
        n_angles = np.size(angle)
    else:
        n_angles = 1

    for i, r in enumerate(radii):

        # --------------------------------------------------------
        # Indices corresponding to this radius
        # --------------------------------------------------------

        start = i * n_angles
        end = start + n_angles

        # --------------------------------------------------------
        # Statistics for all positions at this radius
        # --------------------------------------------------------

        mi_values = []
        pc_values = []

        for j in range(start, end):

            left = left_int[j]
            sym = symmetric_int[j]

            mi = mutual_information(
                left,
                sym,
            )

            pc = pearson_correlation(
                left,
                sym,
            )

            mi_values.append(mi)
            pc_values.append(pc)

        # --------------------------------------------------------
        # Convert to arrays
        # --------------------------------------------------------

        mi_values = np.asarray(mi_values)
        pc_values = np.asarray(pc_values)

        intensity1_values = np.asarray(left_int[start:end])

        intensity2_values = np.asarray(symmetric_int[start:end])

        # --------------------------------------------------------
        # Store mean and standard deviation
        # --------------------------------------------------------

        title, _, _ = reorder_name(name + extraname)
        results.append(
            {
                "name": title,
                "radius": r,
                "intensity1_mean": np.mean(intensity1_values),
                "intensity1_std": np.std(intensity1_values),
                "intensity2_mean": np.mean(intensity2_values),
                "intensity2_std": np.std(intensity2_values),
                "mutual_information_mean": np.mean(mi_values),
                "mutual_information_std": np.std(mi_values),
                "pearson_correlation_mean": np.mean(pc_values),
                "pearson_correlation_std": np.std(pc_values),
            }
        )

    # results_df = pd.DataFrame(results)

    # results_df.insert(
    #     0,
    #     "name",
    #     name + extraname,
    # )

    return results


def make_grid_masks(shape, grid_size=12, include_partial=False, mask_grid=None):
    """
    Create square masks covering an image.

    Parameters
    ----------
    shape : tuple
        Image shape, e.g. (256, 256).
    grid_size : int
        Size of each square in pixels.
    include_partial : bool
        If True, include partial squares at the image edges.
        If False, only complete squares are included.

    Returns
    -------
    masks : dict
        Dictionary of boolean masks.
    """
    ny, nx = shape

    if mask_grid is not None:
        mask = np.asarray(mask_grid, dtype=bool)

        if mask_grid.shape != shape:
            raise ValueError(
                f"Mask shape {mask_grid.shape} does not match " f"image shape {shape}"
            )

    masks = {}

    y_starts = range(0, ny, grid_size)
    x_starts = range(0, nx, grid_size)

    for y in y_starts:
        for x in x_starts:

            y_end = min(y + grid_size, ny)
            x_end = min(x + grid_size, nx)

            # Skip incomplete edge regions if requested
            if not include_partial:
                if (y_end - y < grid_size) or (x_end - x < grid_size):
                    continue

            mask = np.zeros((ny, nx), dtype=bool)
            mask[y:y_end, x:x_end] = True

            name = f"y{y}:{y_end}_x{x}:{x_end}"
            masks[name] = mask

    return masks


def temporal_correlation(
    stack,
    masks=None,
    sub="mean",
    t_max=100,
    grid_size=12,
    mask_grid=None,
    include_partial=False,
    fit_sigma=False,
):
    results = {}

    stack = np.asarray(stack)

    if sub == "median":
        stack = stack - np.median(stack, axis=0)
    elif sub == "mean":
        stack = stack - np.mean(stack, axis=0)

    # Generate grid masks if requested
    if grid_size is not None:
        masks = make_grid_masks(
            stack.shape[1:],
            grid_size=grid_size,
            include_partial=include_partial,
            mask_grid=mask_grid,
        )

    # No masks → full image
    elif masks is None:
        masks = {"full": np.ones(stack.shape[1:], dtype=bool)}

    for name, mask_area in masks.items():
        # Extract only pixels in this region
        X = stack[:, mask_area]

        correlation_matrix = np.full((len(X), len(X)), np.nan)

        corr = np.corrcoef(X)

        i, j = np.triu_indices(len(X), k=1)
        correlation_matrix[i, j] = corr[i, j]

        corr_values = correlation_matrix[~np.isnan(correlation_matrix)]
        hist, bin_edges = np.histogram(corr_values, bins=100)
        median_corr = np.median(corr_values)

        # 1D
        side = correlation_matrix.shape[0]

        correlation_1d_med = np.array(
            [
                np.nanmedian(np.diagonal(correlation_matrix, offset=d))
                for d in range(1, side)
            ]
        )

        correlation_mean = np.array(
            [
                np.nanmean(np.diagonal(correlation_matrix, offset=d))
                for d in range(1, side)
            ]
        )

        correlation_std = np.array(
            [
                np.nanstd(np.diagonal(correlation_matrix, offset=d))
                for d in range(1, side)
            ]
        )

        # Initial guess
        d = np.arange(1, t_max + 1)

        # Restrict to the first `top` distances, then keep only finite values
        mean = correlation_mean[:t_max]
        std = correlation_std[:t_max]

        valid = np.isfinite(mean) & np.isfinite(std) & (std > 0)

        d_fit = d[valid]
        mean_fit = mean[valid]
        std_fit = std[valid]

        # Initial guess
        p0 = [
            mean_fit[0] - mean_fit[-1],  # amplitude
            t_max / 5,  # decay length
            mean_fit[-1],  # baseline
        ]

        # Fit
        params, covariance = curve_fit(
            exp_decay,
            d_fit,
            mean_fit,
            sigma=std_fit if fit_sigma is True else None,
            absolute_sigma=True if fit_sigma is True else False,
            p0=p0,
            maxfev=10000,
        )

        param_std = np.sqrt(np.diag(covariance))

        results[name] = {
            "correlation_matrix_side": side,
            "correlation_histogram": (hist, bin_edges),
            "correlation_median": median_corr,
            "correlation_1d_median": correlation_1d_med,
            "correlation_1d_mean": correlation_mean,
            "correlation_1d_std": correlation_std,
            "exp_fit_params": params,
            "exp_fit_std": param_std,
        }

        # to plot a hist:
    # hist, bin_edges = np.histogram(x, bins=20)

    # widths = np.diff(bin_edges)

    # plt.bar(
    #     bin_edges[:-1],
    #     hist,
    #     width=widths,
    #     align="edge",
    #     edgecolor="black"
    # )

    # plt.show()

    return results
