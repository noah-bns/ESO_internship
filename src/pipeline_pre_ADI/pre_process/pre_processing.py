
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


import pipeline_pre_ADI.library.statistics as stat
importlib.reload(stat)
from pipeline_pre_ADI.library.statistics import *
import pipeline_pre_ADI.library.centering 
importlib.reload(pipeline_pre_ADI.library.centering)
from pipeline_pre_ADI.library.centering  import *
# import utils.plot_pre_processing as plot
# importlib.reload(plot)
# from utils.plot_pre_processing import *

def align_stack(images: np.ndarray,
                    upsample_factor: int = 100,
                    dtype=np.float32,
                    ) -> np.ndarray:

    """
    Align a stack of images using phase cross-correlation + Fourier shift.

    Parameters
    ----------
    images : ndarray
        Image stack with shape (N, ny, nx).

    upsample_factor : int
        Subpixel precision.
        10 -> approximately 0.1 pixel precision.

    Returns
    -------
    aligned : ndarray
        Aligned image stack.

    """
    reference = np.median(images, axis=0)

    aligned = np.empty_like(images, dtype=dtype)

    #Use fourier shift when you want minimum interpolation artifacts (preserve pixel values/speckle statistics)

    for i, image in enumerate(images):

        shift_estimate, _, _ = (
            phase_cross_correlation(
                reference,
                image,
                upsample_factor=upsample_factor, #consider on ROI only
            )
        )

        # aligned[i] = shift(
        #     image,
        #     shift=shift_estimate,
        #     order=3,          #Spatial spline interpolation (order=3) blurs high-frequency speckles. For scientific photometry/PSF characterization, prefer Fourier shifts to preserve pixel values/speckle statistics.
        #     mode="constant",  #mode="constant" + cval=0 introduces black edges that can bias subsequent centroiding. Consider "reflect" or mask edges, or trim a border after shifting.
        #     cval=0.0,
        #     prefilter=True,
        # )

        # apply Fourier shift for minimal smoothing
        image_fft = fftn(image)
        shifted = ifftn(fourier_shift(image_fft, shift_estimate)).real
        aligned[i] = shifted.astype(dtype)

    return aligned 





def estimate_center(
    stack,
    reference_stack=None,
    method="masked_annulus",
    coords=None,
    search_radius=7,
    smooth_sigma=3,
    upsample_factor=100,
    max_iter=100,
    tol=1e-2,
):
    """
    Estimate the center of a stack using the specified method.

    Parameters
    ----------
    stack : ndarray
        Image stack whose center is to be estimated (N, ny, nx).

    reference_stack : ndarray or str, optional
        Stack (or path prefix) of uncorrelated PSFs. If a str is provided,
        files matching f"{reference_stack}*.tif" are read and median-combined.

    method : str
    Estimate center of stack. Supported methods:
      - "cross_correlation" (iterative rotational method)
      - "center_of_mass" (median image centroid, optional mask)
      - "gaussian" (2D rotated gaussian fit on median)
      - "masked_annulus" (annular masked cross-correlation, useful for coronagraphic images)
      - "fixed"

    coords : tuple, optional
        Fixed center (yc, xc), required when method="fixed".

    search_radius : int or None
        Radius (pixels) of the allowed search region around the reference peak.
        If None, no constraint is applied.

    smooth_sigma : float
        Gaussian smoothing sigma when locating the reference PSF peak.

    upsample_factor, max_iter, tol :
        Passed to the cross-correlation routine if method == "cross_correlation".

    Returns
    -------
    yc, xc : float
        Estimated subpixel center in full-image coordinates.
    """

    # validate stack
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("stack must be a 3D array with shape (N, ny, nx)")

    ny, nx = stack.shape[1], stack.shape[2]

    method = method.lower().strip()

    # prepare reference PSF image if provided
    def _prepare_ref(ref):
        if ref is None:
            return None
        if isinstance(ref, str):
            if tiff is None:
                raise RuntimeError("tifffile required to load reference_stack from a path string")
            files = glob.glob(f"{ref}*.tif")
            if len(files) == 0:
                raise FileNotFoundError(f"No files matching '{ref}*.tif'")
            ref_arr = tiff.imread(files)
            ref_m = np.median(ref_arr, axis=0)
        else:
            ref_arr = np.asarray(ref)
            ref_m = np.median(ref_arr, axis=0) if ref_arr.ndim == 3 else ref_arr
        return gaussian_filter(ref_m, smooth_sigma) if (smooth_sigma and smooth_sigma > 0) else ref_m

    ref_psf = _prepare_ref(reference_stack)

    # ---------------------------
    # cross_correlation method
    # ---------------------------
    if method == "cross_correlation":
        # delegate to your existing iterative routine if available
        # this preserves the specialized rotation-based algorithm and parameters
        try:
            yc, xc = estimate_center_correlation(
                stack,
                nframes=min(1000, stack.shape[0]),
                upsample_factor=upsample_factor,
                max_iter=max_iter,
                tol=tol,
                ref_psf = ref_psf,
                search_radius=search_radius,
            )
            return float(yc), float(xc)
        except NameError:
            raise RuntimeError("estimate_center_correlation is not defined in this namespace")

    # ---------------------------
    # fixed method
    # ---------------------------
    if method == "fixed":
        if coords is None:
            raise ValueError("coords must be supplied when method='fixed'.")
        yc, xc = coords
        return float(yc), float(xc)

    if method == "center_of_mass":
        weights = np.median(stack, axis=0)

        if ref_psf is not None:     
            # Approximate PSF peak
            yc0, xc0 = np.unravel_index(
                np.argmax(ref_psf),
                ref_psf.shape,
            )

            if search_radius is None:
                y0, y1, x0, x1 = 0, ny, 0, nx
            else:
                y0 = max(0, int(np.floor(yc0 - search_radius)))
                y1 = min(ny, int(np.ceil(yc0 + search_radius)) + 1)
                x0 = max(0, int(np.floor(xc0 - search_radius)))
                x1 = min(nx, int(np.ceil(xc0 + search_radius)) + 1)
            window = weights[y0:y1, x0:x1]
            window = np.nan_to_num(window - np.nanmin(window))
            if window.sum() <= 0:
                return float((ny - 1) / 2.0), float((nx - 1) / 2.0)
            dy, dx = centroid_com(window)
            yc = float(y0 + dy)
            xc = float(x0 + dx)
            if search_radius is not None:
                yc = float(np.clip(yc, yc0 - search_radius, yc0 + search_radius))
                xc = float(np.clip(xc, xc0 - search_radius, xc0 + search_radius))
            return yc, xc
        else:
            weights = weights - np.nanmin(weights)
            return centroid_com(weights)

        
    if method == "gaussian":
            med = np.median(stack, axis=0).astype(float)
            if ref_psf is not None and search_radius is not None:
                yc0, xc0 = np.unravel_index(np.argmax(ref_psf), ref_psf.shape)
                y0 = max(0, int(np.floor(yc0 - search_radius)))
                y1 = min(ny, int(np.ceil(yc0 + search_radius)) + 1)
                x0 = max(0, int(np.floor(xc0 - search_radius)))
                x1 = min(nx, int(np.ceil(xc0 + search_radius)) + 1)
                bbox = (y0, y1, x0, x1)
            else:
                bbox = None
            
            yc, xc, _ = fit_2d_gaussian(med, bbox=bbox)
            if ref_psf is not None and search_radius is not None:
                yc = float(np.clip(yc, yc0 - search_radius, yc0 + search_radius))
                xc = float(np.clip(xc, xc0 - search_radius, xc0 + search_radius))
            return float(yc), float(xc)


    if method == "masked_annulus":
        # Need a center to build annulus. If mask provided use centroid; if reference PSF available use its peak
        # if mask is not None:
        #     cy, cx = centroid_com(np.median(stack, axis=0), mask=mask)
        cy, cx, _, _ = detect_coron_mask_and_fit_circle(np.median(stack, axis = 0))
        return float(cy), float(cx)
    
    raise ValueError(
        f"Unknown center method: {method}"
    )


def crop_stack(stack, 
               reference_stack=None,
               method="masked_annulus",
               coords=None,
               search_radius=7,
               smooth_sigma=3,
               upsample_factor=100,
               max_iter=100,
               tol=1e-2,
               radius: Optional[int]=None,
               ):
    """
    Crop every frame to the largest odd square centered on (yc, xc).

    Parameters
    ----------
    stack : ndarray
        Image stack, shape (N, ny, nx).

    method : str
        Center estimation method.

        Options:
            "cross_correlation"
            "center_of_mass"
            "fixed"

    coords : tuple, optional
        Fixed center (yc, xc), required when
        method="fixed".
    
    """
    # align the stack
    aligned_stack = align_stack(stack)

    yc, xc = estimate_center(
        stack=aligned_stack, 
        reference_stack=reference_stack, 
        method=method, 
        coords=coords,
        search_radius=search_radius,
        smooth_sigma=smooth_sigma,
        upsample_factor=upsample_factor,
        max_iter=max_iter,
        tol=tol,
        )
    
    ny, nx = aligned_stack.shape[1:]

    center_y = int(np.floor(float(yc) + 0.5))
    center_x = int(np.floor(float(xc) + 0.5))

    # Shift required to move (cy, cx) -> (target_y, target_x)
    dy = center_y - yc
    dx = center_x - xc

    aligned_shifted = shift(
        aligned_stack,
        shift=(0, dy, dx),
        order=3,
        mode="constant",
        cval=0,
    )

    if radius is None:
        half = int(min(center_y, center_x, ny - 1 - center_y, nx - 1 - center_x))
    else:
        half = int(radius)

    cropped = aligned_shifted[
        :,
        center_y - half : center_y + half + 1,
        center_x - half : center_x + half + 1
    ]

    return cropped, (yc, xc)

def zoom_to_peak(img, radius):
    coords_maxpsf = np.unravel_index(np.argmax(img, axis=None), img.shape)
    img = img[
        coords_maxpsf[0] - radius : coords_maxpsf[0] + radius + 1,
        coords_maxpsf[1] - radius : coords_maxpsf[1] + radius + 1,
    ]
    return img

def run_pre_processing(img_file, 
                       img_key = None, 
                       dark_file = None, 
                       dark_key= None, 
                       ref_psf=None, 
                       method = 'masked_annulus',
                       radius = None,
                       coords= None,
                       normalisation = True, 
                       search_radius=7,
                       smooth_sigma=3,
                       upsample_factor=100,
                       max_iter=100,
                       tol=1e-2,
                       eps = 1e-9):
    
    sci_img = tiff.imread(glob.glob(f"{img_file}*.tif"), 
                          key=range(img_key) if img_key else None
                          )
    sci_img_cropped, (yc, xc) = crop_stack(
        sci_img, 
        reference_stack=ref_psf, 
        method=method, 
        radius = radius, 
        coords=coords,
        search_radius=search_radius,
        smooth_sigma=smooth_sigma,
        upsample_factor=upsample_factor,
        max_iter=max_iter,
        tol=tol,
        )

    if dark_file:
        df_img = tiff.imread(glob.glob(f"{dark_file}*.tif"), 
                        key=range(dark_key) if dark_key else None
                        )
        df = crop_stack(df_img, radius = radius, method = 'fixed', coords=(yc, xc))[0]
        master_df = np.median(df, axis = 0)
        sci_img_cropped = (sci_img_cropped - master_df) 

    sci_img_cropped = np.maximum(sci_img_cropped, eps)
    sci_img_processed  = sci_img_cropped.astype(np.float32)

    normalisation_cnst = 1

    if normalisation== True:
        normalisation_cnst = np.max(sci_img_processed)

    sci_img_processed_norm  = sci_img_processed / normalisation_cnst

    return sci_img_processed_norm, normalisation_cnst


# Speckle and outlier analysis

def pca_frame_selection(
    frames,
    n_components=5,
    mad_threshold=5.0,
):
    """
    Perform PCA-based bad frame rejection.

    Parameters
    ----------
    frames : ndarray
        Array of shape (n_frames, height, width).
    n_components : int, optional
        Number of PCA components to compute.
    mad_threshold : float, optional
        Threshold in Median Absolute Deviations (MAD).

    Returns
    -------
    good_mask : ndarray
        Boolean array indicating good frames.
    bad_mask : ndarray
        Boolean array indicating rejected frames.
    pca_coords : ndarray
        Coordinates of each frame in PCA space
        (shape: n_frames × n_components).
    medians : ndarray
        Median of each PCA component.
    mads : ndarray
        MAD of each PCA component.
    """

    n_frames = frames.shape[0]

    # Flatten frames
    X = frames.reshape(n_frames, -1).astype(float)

    # Mean-center each pixel across the dataset
    X -= X.mean(axis=0)

    # PCA (SVD solver)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)

    pca_coords = pca.fit_transform(X)

    # Normalise
    pca_coords /= np.max(np.abs(pca_coords), axis=0)

    # Median and MAD along each PCA axis
    medians = np.median(pca_coords, axis=0)
    mads = np.median(np.abs(pca_coords - medians), axis=0)

    # Avoid division by zero
    mads[mads == 0] = 1e-10

    # Outlier detection
    deviation = np.abs(pca_coords - medians)

    outlier_matrix = deviation > mad_threshold * mads

    bad_mask = np.any(outlier_matrix, axis=1)
    good_mask = ~bad_mask

    return good_mask, bad_mask, outlier_matrix, good_mask, bad_mask, pca_coords, medians, mads


def plot_pca_frame_selection(
    pca_coords,
    bad_mask,
    medians,
    mads,
    outlier_matrix,
    mad_threshold=5,
    figsize=(7, 7),
):
    """
    Plot first two PCA components with MAD rejection regions.

    Parameters
    ----------
    pca_coords : ndarray
        PCA coordinates (n_frames × n_components).
    bad_mask : ndarray
        Boolean mask of rejected frames.
    medians : ndarray
        Median along each PCA component.
    mads : ndarray
        MAD along each PCA component.
    mad_threshold : float
        MAD rejection threshold.
    """

    fig, ax = plt.subplots(figsize=figsize)
    markers = ["D", "s", "^", "P"]
    colors = ['tab:orange', 'tab:pink']

    # Good frames
    ax.scatter(
        pca_coords[~bad_mask, 0],
        pca_coords[~bad_mask, 1],
        s=20,
        color="tab:blue",
        alpha=0.7,
        label=f"Accepted: {np.sum(~bad_mask)} frames",
    )

    # Bad frames
    for pc in range(outlier_matrix.shape[1]):
        mask = outlier_matrix[:, pc]
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            marker='o' if pc < 2 else markers[pc % len(markers)],
            s= 30,
            color= colors[pc] if pc<2 else "black",
            label=f"Outliers in PC{pc+1}: {np.sum(mask)} frames",
        )


    # Rejection rectangle
    xmin = medians[0] - mad_threshold * mads[0]
    xmax = medians[0] + mad_threshold * mads[0]
    ymin = medians[1] - mad_threshold * mads[1]
    ymax = medians[1] + mad_threshold * mads[1]

    ax.axvspan(xmin, xmax, ymin=0, ymax=1,
               color="lightgray", alpha=0.3)
    ax.axhspan(ymin, ymax, xmin=0, xmax=1,
               color="lightgray", alpha=0.3, label = f'{mad_threshold}MAD')

    ax.axvline(medians[0], color="gray", ls="--")
    ax.axhline(medians[1], color="gray", ls="--")

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("\n\nPCA-based Bad Frame Detection")
    ax.legend()

    plt.tight_layout()
    return fig, ax



def run_outlier_detection(img_file, 
                       sci_img_processed,
                    #    img_key = None, 
                    #    dark_file = None, 
                    #    dark_key= None, 
                    #    ref_psf=None, 
                    #    radius = None,
                    #    coords= None,
                       n_components=5,
                       mad_threshold=5.0,
                       PCA_analysis = False,
                       PCA_plot_output = '',
                       normalisation =True,
                       mask = None
                       ):
    
    # sci_img_processed  = run_pre_processing_with_outliers(img_file, 
    #                    img_key, 
    #                    dark_file, 
    #                    dark_key, 
    #                    ref_psf, 
    #                    radius,
    #                    coords)
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            mask = mask.to_image(sci_img_processed.shape[-2:])
        sci_img_processed_masked = sci_img_processed * mask
    else:
        sci_img_processed_masked = sci_img_processed

    good_mask, bad_mask, outlier_matrix, good_mask, bad_mask, pca_coords, medians, mads = pca_frame_selection(
        sci_img_processed_masked,
        n_components=n_components,
        mad_threshold=mad_threshold)

    sci_img_processed_no_outlier = sci_img_processed[good_mask]

    normalisation_cnst = 1
    if normalisation == True:
        normalisation_cnst =  np.max(sci_img_processed_no_outlier)

    sci_img_processed_no_outlier_norm  = sci_img_processed_no_outlier / normalisation_cnst


    if PCA_analysis==True:
        fig, ax = plot_pca_frame_selection(
            pca_coords,
            bad_mask,
            medians,
            mads,
            outlier_matrix,
            mad_threshold= int(mad_threshold),
        )
        name = os.path.basename(img_file)          # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
        save_name = name.split("_2026")[0] if "_2026" in name else name # int_1.0ro_10ws_no_coro
        plt.suptitle(save_name.replace('_', ' '))
        plt.savefig(f'{PCA_plot_output}{save_name.replace(' ', '_')}.png', dpi=300, bbox_inches="tight")
        plt.show()

    return sci_img_processed_no_outlier_norm, outlier_matrix, normalisation_cnst



def radial_profile_pre(image, binsize=1):
    ny, nx = image.shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2

    y, x = np.indices(image.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    rbin = np.floor(r / binsize).astype(int)

    maxbin = rbin.max() + 1

    profile = np.empty(maxbin)
    std = np.empty(maxbin)
    radius = np.arange(maxbin) * binsize

    for i in range(maxbin):
        mask = rbin == i
        values = image[mask]

        profile[i] = values.mean()
        std[i] = values.std()

    return radius, profile, std




def symmetric_intensities(
    cube,
    angles_deg,
    r_min,
    r_max,
    dr=5.0,
    r_aperture = 0.5,
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
        r_min + 0.5 * dr + r_aperture,
        r_max - r_aperture,
        (dr + 2 * r_aperture)
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
    left = np.empty(
        (n_frames, n_radii * n_angles),
        dtype=cube.dtype
    )

    symmetric = np.empty_like(left)

    # --------------------------------------------------------
    # Extract intensities
    # --------------------------------------------------------
    for i, frame in enumerate(cube):

        tmp_apertures_left = CircularAperture(positions=np.column_stack((x1, y1)),
                                            r=r_aperture)

        left[i, :] = aperture_photometry(
                frame,
                tmp_apertures_left,
                method='center')['aperture_sum']
        
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
        tmp_apertures_sym = CircularAperture(positions=np.column_stack((x2, y2)),
                                            r=r_aperture)

        symmetric[i, :] = aperture_photometry(
                frame,
                tmp_apertures_sym,
                method='center')['aperture_sum']
        
    return (x1,y1), left, (x2,y2), symmetric

