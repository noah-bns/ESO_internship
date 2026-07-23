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
from scipy.ndimage import gaussian_filter
import importlib
import time
import gc
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm

# images
mpl.rcParams["hatch.linewidth"] = 0.5  # previous pdf hatch linewidth
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
from sklearn.decomposition import PCA

# import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter

# scientific libraries
from hcipy import *
import applefy

# importlib.reload(applefy)
from applefy import *

# importlib.reload(applefy.detections.contrast)
# from applefy.detections.contrast import Contrast
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.wrappers.pynpoint import MultiComponentPCAPynPoint
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import (
    TTest,
    gaussian_sigma_2_fpf,
    fpf_2_gaussian_sigma,
    LaplaceBootstrapTest,
)
from applefy.wrappers.vip import MultiComponentPCAvip

import fours

# importlib.reload(fours)
from fours.detection_limits.applefy_wrapper import (
    CADIDataReductionGPU,
    PCADataReductionGPU,
    CADIDataReduction,
)
from fours.models.rotation import FieldRotationModel

from skimage.registration import phase_cross_correlation


def estimate_center(
    stack,
    nframes=1000,
    upsample_factor=100,
    max_iter=100,
    tol=1e-2,
    reference_stack=None,
    search_radius=7,
    smooth_sigma=2,
):

    """
    Estimate the center of a stack using iterative 180° rotational
    phase cross-correlation.

    Parameters
    ----------
    stack : ndarray
        Image stack whose center is to be estimated.

    reference_stack : ndarray, optional
        Stack of uncorrelated PSFs. The median of this stack is used to
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

    ref = np.median(stack[:nframes], axis=0)
    ny, nx = ref.shape

    # ------------------------------------------------------------------
    # Initial center
    # ------------------------------------------------------------------
    if reference_stack is not None:
        if isinstance(reference_stack, str):
            reference_stack = tiff.imread(glob.glob(f"{reference_stack}*.tif"))
        ref_psf = np.median(reference_stack, axis=0)
        ref_psf = gaussian_filter(ref_psf, smooth_sigma)

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

        sub = ref[y - half : y + half + 1, x - half : x + half + 1]

        rot = np.rot90(sub, 2)

        shift, _, _ = phase_cross_correlation(sub, rot, upsample_factor=upsample_factor)

        dy, dx = shift / 2

        yc += dy
        xc += dx

        # Constrain search around reference PSF center
        if reference_stack is not None:
            yc = np.clip(yc, yc0 - search_radius, yc0 + search_radius)
            xc = np.clip(xc, xc0 - search_radius, xc0 + search_radius)


        if np.hypot(dy, dx) < tol:
            print(f"Center converged after {_+1} iterations.")
            break

    return (yc, xc)


def crop_stack(stack, reference_stack=None, coords=None, radius=None):
    """
    Crop every frame to the largest odd square centered on (yc, xc).
    """

    if coords is None:
        yc, xc = estimate_center(stack, reference_stack=reference_stack)

    else:
        yc, xc = coords

    ny, nx = stack.shape[1:]

    if radius is None:
        half = int(min(yc, xc, ny - 1 - yc, nx - 1 - xc))
    else:
        half = radius

    yc = int(round(yc))
    xc = int(round(xc))

    return stack[:, yc - half : yc + half + 1, xc - half : xc + half + 1]


def zoom_to_peak(img, radius):
    coords_maxpsf = np.unravel_index(np.argmax(img, axis=None), img.shape)
    img = img[
        coords_maxpsf[0] - radius : coords_maxpsf[0] + radius + 1,
        coords_maxpsf[1] - radius : coords_maxpsf[1] + radius + 1,
    ]
    return img

def run_pre_processing_with_outliers(img_file, 
                       img_key = None, 
                       dark_file = None, 
                       dark_key= None, 
                       ref_psf=None, 
                       radius = None,
                       coords= None,
                       ):
    
    sci_img = tiff.imread(glob.glob(f"{img_file}*.tif"), 
                          key=range(img_key) if img_key else None
                          )
    sci_img_cropped = crop_stack(sci_img,  radius = radius, coords=coords)

    if dark_file:
        df_img = tiff.imread(glob.glob(f"{dark_file}*.tif"), 
                        key=range(dark_key) if dark_key else None
                        )
        df = crop_stack(df_img, radius = radius, reference_stack = ref_psf, coords=coords)
        master_df = np.median(df, axis = 0)
        sci_img_cropped = (sci_img_cropped - master_df) 

    sci_img_processed  = sci_img_cropped.astype(np.float32) / 65535
    return sci_img_processed


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



def run_pre_processing(img_file, 
                       img_key = None, 
                       dark_file = None, 
                       dark_key= None, 
                       ref_psf=None, 
                       radius = None,
                       coords= None,
                       n_components=5,
                       mad_threshold=5.0,
                       PCA_analysis = False
                       ):
    
    sci_img_processed  = run_pre_processing_with_outliers(img_file, 
                       img_key, 
                       dark_file, 
                       dark_key, 
                       ref_psf, 
                       radius,
                       coords)

    good_mask, bad_mask, outlier_matrix, good_mask, bad_mask, pca_coords, medians, mads = pca_frame_selection(
        sci_img_processed,
        n_components=n_components,
        mad_threshold=mad_threshold)

    sci_img_processed_no_outlier = sci_img_processed[good_mask]

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
        save_name = name.split("_2026")[0]  # int_1.0ro_10ws_no_coro
        plt.suptitle(save_name.replace('_', ' '))
        plt.savefig(f'{save_name.replace(' ', '_')}.png', dpi=300, bbox_inches="tight")
        plt.show()
    return sci_img_processed_no_outlier, outlier_matrix


def plot_outlier_images_by_pc(
    cube,
    outlier_matrix,
    cmap="inferno",
):
    """
    Plot all outlier images grouped by PCA component.

    Parameters
    ----------
    cube : ndarray
        Image cube (n_frames, ny, nx)
    outlier_matrix : ndarray
        Boolean array (n_frames, n_components)
    """

    n_components = outlier_matrix.shape[1]

    # Global normalization over ALL outlier images
    all_indices = np.where(np.any(outlier_matrix, axis=1))[0]

    if len(all_indices) == 0:
        print("No outliers found.")
        return

    data = cube[all_indices]

    # Log scale requires positive values
    vmin = np.percentile(data[data > 0], 1)
    vmax = np.percentile(data[data > 0], 99.5)

    norm = LogNorm(vmin=vmin, vmax=vmax)


    for pc in range(n_components):

        indices = np.where(outlier_matrix[:, pc])[0]

        if len(indices) == 0:
            continue

        ncols = 6
        nrows = int(np.ceil(len(indices) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3*ncols, 3*nrows),
        )

        axes = np.atleast_1d(axes).ravel()

        im = None

        for ax, idx in zip(axes, indices):

            # avoid zero values for LogNorm
            image = np.maximum(cube[idx], vmin)

            im = ax.imshow(
                image,
                origin="lower",
                cmap=cmap,
                norm=norm,
            )

            ax.set_title(f"Frame {idx}")
            ax.axis("off")


        # remove unused panels
        for ax in axes[len(indices):]:
            ax.remove()


        # one colorbar for this PC group
        cbar = fig.colorbar(
            im,
            ax=axes[:len(indices)],
            fraction=0.02,
            pad=0.02,
        )

        cbar.set_label("Intensity (log scale)")


        fig.suptitle(
            f"Outliers detected by PC{pc+1} "
            f"({len(indices)} frames)",
            fontsize=16,
        )

        plt.show()


if __name__ == "__main__":
    
    save_path = '/home/aosimul/noah/data/ghost_images/10_20ws/'

    ori_folder = '/home/aosimul/noah/data/ghost_images/phase_screens_SLM/7_22/'

    # Get all files
    files = sorted(glob.glob(f"{ori_folder}*.tif"))

    # Separate correctly
    img_files = [
        f for f in files
        if "coro" in os.path.basename(f)
        and "no_coro" not in os.path.basename(f)
    ]

    psf_files = [
        f for f in files
        if "no_coro" in os.path.basename(f)
    ]


    print(f"Science images: {len(img_files)}")
    print(f"PSF images: {len(psf_files)}")

    for img_file in img_files:
        sci_img_processed_no_outlier, outlier_matrix = run_pre_processing(img_file = ori_folder + Path(img_file).stem, 
                            img_key = 6667, 
                            dark_file = f'{ori_folder}darks',  
                            dark_key= 8475, 
                            ref_psf=f'{ori_folder}psf', 
                            radius = 55,
                            n_components=3,
                            mad_threshold=7,
                            PCA_analysis = True
                            )
        name = os.path.basename(img_file)          # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
        save_name = name.split("_coro")[0]  # int_1.0ro_10ws_no_coro
        
        sci_img_processed = run_pre_processing_with_outliers(img_file = ori_folder + Path(img_file).stem,
                        img_key = 6667, 
                        dark_file = f'{ori_folder}darks',  
                        dark_key= 8475, 
                        ref_psf=f'{ori_folder}psf', 
                        radius = 55,
                        coords= None
                        )
        plot_outlier_images_by_pc(
            sci_img_processed, outlier_matrix
            )
        np.save(f'{save_path}{save_name}.npy', sci_img_processed_no_outlier)

    for psf_file in psf_files:
        psf_img_processed = run_pre_processing_with_outliers(img_file = ori_folder + Path(psf_file).stem, 
                            img_key = 8475, 
                            dark_file = f'{ori_folder}darks',  
                            dark_key= 8475, 
                            ref_psf=f'{ori_folder}psf', 
                            radius = 55
                            )
        psf_med = np.sum(psf_img_processed, axis = 0)
        name = os.path.basename(psf_file)          # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
        save_name = name.split("_no")[0]  # int_1.0ro_10ws_no_coro
        np.save(f'{save_path}psf_{save_name}.npy', psf_med)