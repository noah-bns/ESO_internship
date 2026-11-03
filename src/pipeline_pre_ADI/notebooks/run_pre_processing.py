from pipeline_pre_ADI.pre_process.pre_processing import *
from pipeline_pre_ADI.pre_process.plot_pre_processing import *
from utils.naming import reorder_name

# org
import shutil
import torch
import numpy as np
import warnings
import os, re, glob
from scipy import interpolate
import pandas as pd
from copy import deepcopy
from collections import Counter
import h5py, json, math

# images
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile as tiff
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import center_of_mass
from photutils.aperture import CircularAperture

plt.close("all")


# scientific libraries
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.utils.photometry import AperturePhotometryMode
from applefy.detections.contrast import Contrast
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, LaplaceBootstrapTest
from applefy.utils.positions import center_subpixel

from pathlib import Path

root_dir = Path(".")
print(root_dir)


# MODE
create = True
counting = True
analysis = False
start_over = False
compute_raw_contrast = False

# LOAD FILES

save_data_path = "/home/aosimul/noah/data/ghost_images/long_7ro_20ws_focus"
if start_over == True:
    if os.path.exists(save_data_path):
        shutil.rmtree(Path(save_data_path))
        print(f"Removed existing directory to avoid overwrites: {save_data_path}.")

if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

save_analysis_path = (
    "/home/aosimul/noah/src/pipeline_pre_ADI/results/long_7ro_20ws_focus"
)
ori_folder = "/home/aosimul/noah/data/ghost_images/phase_screens_SLM/10_10/"

if not os.path.exists(save_analysis_path):
    os.makedirs(save_analysis_path + "/speckle_diversity")

# geometry
radius = 55
mask_radius = 35
mask_outlier = CircularAperture((radius, radius), mask_radius).to_mask(method="center")
DIT_psf = 29e-6
DIT_img = 2.814e-3
img_key = None

# Get all files
files = sorted(glob.glob(f"{ori_folder}*.tif"))

# Separate correctly
img_files = [
    f
    for f in files
    if "coro" in os.path.basename(f) and "no_coro" not in os.path.basename(f)
]

psf_files = [f for f in files if "no_coro" in os.path.basename(f)]

# Count occurrences of each root
img_root_counts = Counter(reorder_name(os.path.basename(f))[1] for f in img_files)

psf_root_counts = Counter(reorder_name(os.path.basename(f))[1] for f in psf_files)
# Running counters while saving
img_counter = Counter()
psf_counter = Counter()

print(f"Science images: {len(img_files)}")
print(f"PSF images: {len(psf_files)}")


# CREATE FILES
psf_count = 0
diffs_outliers_imgs = {}
diffs_outliers_psf = {}
light_intensities = {}

# 0) Save an uncorrelated median psf
if compute_raw_contrast == True:
    psf_uncorelated, norm_const_uncorelated, _ = run_pre_processing(
        img_file=ori_folder + "psf_",
        img_key=10000,
        dark_file=f"{ori_folder}darks",
        dark_key=10000,
        # ref_psf=f'{ori_folder}psf_',
        radius=radius,
        normalisation=True,
        method="waffle",
        n_satellites=2,
    )

    psf_med = np.mean(psf_uncorelated, axis=0)
    if create == True:
        np.save(f"{save_data_path}/psf_uncorrelated.npy", psf_med)


for i, img_file in enumerate(img_files[:]):
    fwhm = None
    # 1) Align, center and crop
    sci_img_processed, norm_const_1, speckle_intensity = run_pre_processing(
        img_file=ori_folder + Path(img_file).stem,
        img_key=img_key,
        dark_file=f"{ori_folder}darks_img",
        dark_key=10000,
        # ref_psf=f'{ori_folder}psf_',
        radius=radius,
        method="waffle",
        n_satellites=2,
        normalisation=False,
    )

    # 2) Sort filename
    name = os.path.basename(
        img_file
    )  # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
    _, save_name, _ = reorder_name(name)  # int_10ro_10ws

    if img_root_counts[save_name] > 1:
        img_counter[save_name] += 1
        filename = f"{save_name}_{img_counter[save_name]}.npy"
    else:
        filename = f"{save_name}.npy"

    print(img_file)
    print(save_name)
    print(filename)

    # 3) Detect outliers
    sci_img_processed_no_outlier, outlier_matrix, norm_const_2 = run_outlier_detection(
        stack_name=filename.split(".npy")[0],
        sci_img_processed=sci_img_processed,
        n_components=3,
        mad_threshold=6,
        PCA_analysis=True,
        PCA_plot_output=save_analysis_path + "/speckle_diversity",
        normalisation=False,
        mask=mask_outlier,
    )

    # plot_outlier_images_by_pc(
    #     sci_img_processed, outlier_matrix
    #     )

    # 4) Save the file
    mean_sci_img_processed_no_outlier = np.mean(sci_img_processed_no_outlier, axis=0)
    diff = np.mean(sci_img_processed, axis=0) - mean_sci_img_processed_no_outlier
    diffs_outliers_imgs[filename.split(".npy")[0]] = diff

    if compute_raw_contrast == True:
        pixel_rad, y_noncoro, _ = radial_profile_pre(mean_sci_img_processed_no_outlier)
        raw_contrast_amplitude = (
            norm_const_1 / norm_const_2 / norm_const_uncorelated * DIT_psf / DIT_img
        )

    if create == True:
        np.save(f"{save_data_path}/{filename}", sci_img_processed_no_outlier)

    # 5) Do the same if psf corresponds
    if psf_files:
        mod = i % 2
        if mod != 1:
            # 5.1) Check if need to do psf
            psf_file = psf_files[psf_count]
            psf_img_processed, _, _ = run_pre_processing(
                img_file=ori_folder + Path(psf_file).stem,
                img_key=10000,
                dark_file=f"{ori_folder}darks_psf",
                dark_key=10000,
                # ref_psf=f'{ori_folder}psf_',
                radius=radius,
                method="waffle",
                n_satellites=2,
                normalisation=False,
            )
            # 5.2) Detect outliers
            # psf_img_processed_no_outlier, outlier_matrix = run_outlier_detection(img_file = ori_folder + Path(psf_file).stem,
            #                     sci_img_processed = psf_img_processed,
            #                     n_components=3,
            #                     mad_threshold=6,
            #                     PCA_analysis = False,
            #                     PCA_plot_output= save_analysis_path + '/speckle_diversity',
            #                     normalisation= True
            #                     )
            # plot_outlier_images_by_pc(
            #     psf_img_processed, outlier_matrix
            #     )
            # diff = np.mean(psf_img_processed, axis = 0) - np.mean(psf_img_processed_no_outlier, axis = 0)

            # 5.3) Save the file
            psf_med = np.mean(psf_img_processed, axis=0)
            fwhm = calculate_fwhm(psf_med)

            name = os.path.basename(
                psf_file
            )  # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
            _, save_name, _ = reorder_name(name)  # int_10ro_10ws

            if psf_root_counts[save_name] > 1:
                psf_counter[save_name] += 1
                filename_psf = f"psf_{save_name}_{img_counter[save_name]}.npy"
            else:
                filename_psf = f"psf_{save_name}.npy"

            # name = os.path.basename(psf_file)          # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
            # save_name = name.split("_no")[0]  # int_1.0ro_10ws_no_coro
            # np.save(f'{save_path}/psf_{save_name}.npy', psf_med)

            if create == True:
                np.save(f"{save_data_path}/{filename_psf}", psf_med)
            # diffs_outliers_psf[filename.split('.n')[0]] = (diff)

            psf_count += 1

    if compute_raw_contrast == True:
        raw_contrast = pd.DataFrame(
            {
                "raw_contrast": raw_contrast_amplitude * y_noncoro,
            },
            index=pd.MultiIndex.from_arrays(
                [
                    pixel_rad,
                    (
                        pixel_rad / fwhm
                        if fwhm is not None
                        else np.full_like(pixel_rad, np.nan, dtype=float)
                    ),
                ],
                names=["pixel_rad", "pixel_rad_fwhm"],
            ),
        )
        raw_contrast.to_hdf(
            f"{save_analysis_path}/raw_contrast.h5",
            key=filename.split(".n")[0],
            mode="a",
        )

    plt.close("all")
    print(f"Done: {filename}")

    light_intensities[filename.split(".n")[0]] = {
        # "psf": None if math.isnan(norm_max_PSF) else float(norm_max_PSF),
        # "psf_photometry": None if math.isnan(peak_value_psf) else float(peak_value_psf),
        # "satellite_photometry": None if math.isnan(peak_value) else float(peak_value),
        "satellite": None if math.isnan(speckle_intensity) else float(speckle_intensity)
    }


# Save locally
with open(f"{save_analysis_path}/light_intensities.json", "w") as f:
    json.dump(light_intensities, f, indent=4)


with h5py.File(f"{save_analysis_path}/outlier_diff.h5", "a") as f:
    for key, value in diffs_outliers_imgs.items():
        if key in f:
            del f[key]
        f.create_dataset(key, data=value)
