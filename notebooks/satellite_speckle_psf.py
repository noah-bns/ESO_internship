import math

from pipeline_pre_ADI.pre_process.centering import *
from pipeline_pre_ADI.pre_process.pre_processing import *
from pipeline_pre_ADI.pre_process.plot_pre_processing import *
from utils.statistics import rect_aperture_mask, calculate_fwhm

# org
import shutil
import torch
import numpy as np
import warnings
import os, re, glob
import pandas as pd
from copy import deepcopy
from collections import Counter
from pathlib import Path
import time
import gc
from typing import Optional, Tuple, Sequence
import pickle
import json
import math
from functools import partial

# scientific libraries
from photutils.aperture import aperture_photometry, CircularAperture
from scipy.ndimage import gaussian_filter, shift, map_coordinates, fourier_shift
from skimage.registration import phase_cross_correlation
from applefy.utils.positions import center_subpixel
from numpy.fft import fftn, ifftn
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
from scipy import signal

# images
# import utils.plot_pre_processing as plot
# importlib.reload(plot)
# from utils.plot_pre_processing import *
# images
import matplotlib.pyplot as plt
import tifffile as tiff
from photutils.aperture import CircularAperture

mpl.rcParams["hatch.linewidth"] = 0.5  # previous pdf hatch linewidth
plt.close("all")


from pathlib import Path

root_dir = Path(".")
print(root_dir)

save_data_path = "/home/aosimul/noah/data/ghost_images/10_15_20ws/"

ori_folder = "/home/aosimul/noah/data/ghost_images/phase_screens_SLM/8_24/"


# Get all files
files = sorted(glob.glob(f"{ori_folder}*.tif"))

# Separate correctly
img_files = [
    f
    for f in files
    if "_coro" in os.path.basename(f) and "no_coro" not in os.path.basename(f)
]

psf_files = [f for f in files if "no_coro" in os.path.basename(f)]

# Count occurrences of each root
img_root_counts = Counter(os.path.basename(f).split("_coro")[0] for f in img_files)

psf_root_counts = Counter(os.path.basename(f).split("_no_coro")[0] for f in psf_files)


print(f"Science images: {len(img_files)}")
print(f"PSF images: {len(psf_files)}")


# Running counters while saving
img_counter = Counter()
psf_counter = Counter()
DIT_psf = 29e-6
psf_count = 0
r_ap = 4

light_intensities = {}


for i, img_file in enumerate(img_files[:]):

    # 2) Sort filename
    name = os.path.basename(
        img_file
    )  # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
    _, save_name, _ = reorder_name(name)  # int_10ro_10ws

    if img_root_counts[save_name] > 1:
        img_counter[save_name] += 1
        filename = f"sat_psf_{save_name}_{img_counter[save_name]}.npy"
    else:
        filename = f"sat_psf_{save_name}.npy"

    print(img_file)
    print(save_name)
    print(filename)

    psf = np.load(f"{save_data_path}psf_{save_name}.npy")

    norm_max_PSF = np.max(psf)

    # Do photometry
    peak_value_psf = center_photometry(psf, r_ap)

    sat_img_processed, _ = run_pre_processing(
        img_file=ori_folder + Path(img_file).stem,
        masking_function=rect_mask,  # partial(rect_aperture_mask, x=102, y=172),
        img_key=9986,
        coords=(172, 102),
        dark_file=f"{ori_folder}darks_psf",
        dark_key=10000,
        # ref_psf=f'{ori_folder}psf_',
        radius=10,
        method="fixed",
        n_satellites=2,
        normalisation=False,
    )
    sat_med = np.median(sat_img_processed, axis=0)
    # plt.imshow(sat_med)
    # plt.colorbar()
    # plt.show()
    fwhm = calculate_fwhm(sat_med)
    psf_count += 1
    max_sat = np.max(sat_med)

    # Do photometry
    peak_value = center_photometry(sat_med, r_ap)

    light_intensities[f"{save_name}_{img_counter[save_name]}"] = {
        "psf": None if math.isnan(norm_max_PSF) else float(norm_max_PSF),
        "psf_photometry": None if math.isnan(peak_value_psf) else float(peak_value_psf),
        "satellite_photometry": None if math.isnan(peak_value) else float(peak_value),
        "satellite": None if math.isnan(max_sat) else float(max_sat),
    }


# Save locally
with open("/home/aosimul/noah/notebooks/light_intensities.json", "w") as f:
    json.dump(light_intensities, f, indent=4)
