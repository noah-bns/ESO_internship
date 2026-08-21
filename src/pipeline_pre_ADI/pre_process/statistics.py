
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


def rsquared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot > 0:
        return 1 - ss_res / ss_tot
    else:
        return np.nan





def mutual_information(stack, pos1, pos2):
    """
    Mutual information between two pixel positions across a stack of images.

    Parameters
    ----------
    stack : np.ndarray
        Image stack with shape (N, height, width).
    pos1 : tuple
        (row, col) of the first pixel.
    pos2 : tuple
        (row, col) of the second pixel.

    Returns
    -------
    float
        Mutual information in nats.
    """

    # Extract pixel values across all images
    x = stack[:, pos1[0], pos1[1]]
    y = stack[:, pos2[0], pos2[1]]

    # Round normalized values to 1e-4
    x = np.round(x, 4)
    y = np.round(y, 4)

    # Remove NaN/inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Joint probability distribution
    xy, counts_xy = np.unique(
        np.column_stack((x, y)),
        axis=0,
        return_counts=True,
    )

    p_xy = counts_xy / len(x)

    # Marginal probabilities
    _, counts_x = np.unique(x, return_counts=True)
    _, counts_y = np.unique(y, return_counts=True)

    p_x = counts_x / len(x)
    p_y = counts_y / len(y)

    # MI = sum p(x,y) log[p(x,y)/(p(x)p(y))]
    mi = 0.0

    for (x_val, y_val), p_joint in zip(xy, p_xy):

        p_x_val = p_x[np.where(np.unique(x) == x_val)[0][0]]
        p_y_val = p_y[np.where(np.unique(y) == y_val)[0][0]]

        mi += p_joint * np.log(
            p_joint / (p_x_val * p_y_val)
        )

    return mi



def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two datasets.

    Parameters
    ----------
    x, y : array-like
        Two datasets with the same number of elements.

    Returns
    -------
    float
        Pearson correlation coefficient, between -1 and 1.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    # Remove NaN / inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    # Avoid division by zero for constant arrays
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return np.corrcoef(x, y)[0, 1] #pick either off diagonal element of the correlation matrix



def pixel_P_correlation(stack, pos1, pos2):
    """
    Pearson correlation between two pixel positions
    across a stack of images.
    """

    x = stack[:, pos1[0], pos1[1]]
    y = stack[:, pos2[0], pos2[1]]

    return pearson_correlation(x, y)


# For two images:
# r = pearson_correlation(
#     image1.ravel(),
#     image2.ravel(),
# )