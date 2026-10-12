from pipeline_pre_ADI.pre_process.pre_processing import *
from pipeline_pre_ADI.pre_process.plot_pre_processing import *
from pipeline_pre_ADI.pre_process.correlation import *

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
import pickle, h5py

# images
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile as tiff
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import center_of_mass
from photutils.aperture import CircularAperture, CircularAnnulus

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


ori_folder = "/home/aosimul/noah/data/ghost_images/10_15_20ws/"

shape = (111, 111)
center = (50, 50)

edge_coro = 8
edge_stg1 = 34
edge_stg2 = 60
edge_atm = 160

masks = {
    #'coro' : ~CircularAperture(center, edge_coro).to_mask(method = 'center').to_image(sci_img.shape[-2:]).astype(bool),
    "stage_1": CircularAnnulus(center, edge_coro, edge_stg1)
    .to_mask(method="center")
    .to_image(shape)
    .astype(bool),
    "stage_2": CircularAnnulus(center, edge_stg1, edge_stg2)
    .to_mask(method="center")
    .to_image(shape)
    .astype(bool),
    "rest": CircularAnnulus(center, edge_stg2, edge_atm)
    .to_mask(method="center")
    .to_image(shape)
    .astype(bool),
}
masks.update(
    make_grid_masks(shape, grid_size=10, include_partial=False, mask_grid=None)
)


for algo in ["int", "pred"]:
    for ro in [5, 7, 10]:
        for ws in [10, 15, 20]:
            for count in [1, 2]:
                name, _, _ = reorder_name(f"{algo}_{ro}ro_{ws}ws_{count}")
                sci_img = np.load(f"{ori_folder}{name}.npy")
                results = temporal_correlation(sci_img, masks=masks, sub="mean")
                results_df = pd.DataFrame(results)
                results_df.to_hdf(
                    f"{ori_folder}temporal_correlation.h5", key=name, mode="a"
                )
                print(results_df.shape)
                print(results_df.isna().sum().sum())
                print(np.isinf(results_df.select_dtypes(include=np.number)).sum().sum())

                print("Done: ", name)
