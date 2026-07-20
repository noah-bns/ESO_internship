
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
        radius):
    coords_maxpsf = (np.unravel_index(np.argmax(img, axis=None), img.shape))
    img = img[coords_maxpsf[0]-radius:coords_maxpsf[0]+radius+1, coords_maxpsf[1]-radius:coords_maxpsf[1]+radius+1]
    return img
