import numpy as np
from pathlib import Path
import warnings
import os, glob, re
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


def section_name(s):
    """
    Reorder a name string to have the following order:
    1. priority: "int" or "pred"
    2. arcsec: "<number>r0" or "<number>ro"
    3. windspeed: "<number>windspeed" or "<number>ws"
    4. key: "<number>key" or "key<number>"
    5. everything else in the order they appear

    Output: (string, string, string, string, string)
    """
    parts = s.split("_")

    priority = None
    arcsec = None
    windspeed = None
    key = None
    rest = [""]

    for part in parts:

        # int / pred
        if part in ("int", "pred"):
            priority = part

        # arcsec: <number>r0 or <number>ro
        elif re.fullmatch(r"\d*\.?\d+(?:r0|ro)", part):

            match = re.fullmatch(r"(\d*\.?\d+)(?:r0|ro)", part)

            number_str = match.group(1)

            if "." in number_str:
                number = int(float(number_str) * 10)
            else:
                number = int(number_str)

            arcsec = f"{number}ro"

        # e.g. 10windspeed, 10ws
        elif re.fullmatch(r"\d+(?:windspeed|ws)", part):
            number = re.match(r"\d+", part).group()
            windspeed = f"{int(number)}ws"

        # e.g. 10key, key10
        elif re.fullmatch(r"(?:\d+key|key\d+)", part):
            number = re.search(r"\d+", part).group()
            key = f"{int(number)}key"

        else:
            rest.append(part)

    return priority, arcsec, windspeed, key, rest


def reorder_name(s):
    """
    Reorder a name string to have the following order:
    1. priority: "int" or "pred"
    2. arcsec: "<number>r0" or "<number>ro"
    3. windspeed: "<number>windspeed" or "<number>ws"
    4. key: "<number>key" or "key<number>"
    5. everything else in the order they appear

    Output: (string, string, string)
    - full: the full reordered name
    - special: the special components (priority, arcsec, windspeed, key)
    - rest: the rest of the components
    """

    priority, arcsec, windspeed, key, rest = section_name(s)

    # Put special components first, then everything else
    output = []

    if priority is not None:
        output.append(priority)

    if arcsec is not None:
        output.append(arcsec)

    if windspeed is not None:
        output.append(windspeed)

    if key is not None:
        output.append(key)

    full = output.copy()
    full.extend(rest)

    return (
        "_".join(full),
        "_".join(output),
        "_".join(rest),
    )


def titling(s, key = False):
    _, special, extraname = reorder_name(s)

    if "pred" in special:
        special = special.replace("pred", "Predictive AO:")

    if "int" in special:
        special = special.replace("int", "Integrative AO:")

    # Divide ro value by 10 for the title
    special = re.sub(
        r"(\d+(?:\.\d+)?)ro", lambda m: f"{float(m.group(1)) / 10:g} arcsec,", special
    )

    if "ws" in special:
        special = special.replace("ws", " m/s")

    if key and "key" in special:
        special = special.replace("key", ", Key: ")

    special = special.replace("_", " ")

    extraname = f'\n({extraname.replace("_", " ")} )' if extraname is not "" else ""

    return special + extraname
