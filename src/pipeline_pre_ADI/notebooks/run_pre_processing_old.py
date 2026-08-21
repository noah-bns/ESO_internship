import importlib
import utils.functions as functions 
import pipeline_pre_ADI.library.pre_processing as pre_processing
import pipeline_pre_ADI.library.plot_pre_processing as plot 
importlib.reload(pre_processing)
importlib.reload(functions)
importlib.reload(plot)
from utils.functions import *
from pipeline_pre_ADI.library.pre_processing import *
from pipeline_pre_ADI.library.plot_pre_processing import *

#org
import shutil 
import torch
import numpy as np
import warnings
import os, re, glob
from scipy import interpolate
import pandas as pd
from copy import deepcopy
from collections import Counter
import pickle 

#images
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile as tiff
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import center_of_mass
from photutils.aperture import CircularAperture
plt.close('all')


#scientific libraries
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.utils.photometry import AperturePhotometryMode
from applefy.detections.contrast import Contrast
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, LaplaceBootstrapTest
from applefy.utils.positions import center_subpixel

from pathlib import Path
root_dir = Path(".")
print(root_dir)



# LOAD FILES
path = '/home/aosimul/noah/data/ghost_images/10_20ws'
if os.path.exists(path):
    shutil.rmtree(Path(path))
    print(f"Removed existing directory to avoid overwrites: {path}.")

if not os.path.exists(path):
    os.makedirs(path)

save_path = path + '/'
save_plot_path = '/home/aosimul/noah/results/animations/speckle_diversity/batch_1/'
ori_folder = '/home/aosimul/noah/data/ghost_images/phase_screens_SLM/7_22/'


#geometry
radius = 55
mask_radius = 35
mask_outlier = CircularAperture((radius, radius), mask_radius).to_mask(method = 'center')
DIT_psf = 59e-6
DIT_img = 15e-3

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

# Count occurrences of each root
img_root_counts = Counter(
    os.path.basename(f).split("_coro")[0]
    for f in img_files
)

psf_root_counts = Counter(
    os.path.basename(f).split("_no_coro")[0]
    for f in psf_files
)

# Running counters while saving
img_counter = Counter()
psf_counter = Counter()

print(f"Science images: {len(img_files)}")
print(f"PSF images: {len(psf_files)}")



#CREATE FILES

create = True
psf_count = 0
diffs_outliers_imgs = {}
# diffs_outliers_psf = {}
raw_contrast_amplitudes = {}


# 0) Save an uncorrelated median psf   
psf_uncorelated, norm_const_uncorelated = run_pre_processing(img_file = ori_folder + 'psf_', 
                    img_key = 8475, 
                    dark_file = f'{ori_folder}darks_psf',  
                    dark_key= 10000, 
                    ref_psf=f'{ori_folder}psf_', 
                    radius = radius,
                    normalisation=True
                    )

psf_med = np.mean(psf_uncorelated, axis = 0)
if create == True:
    np.save(f'{save_path}psf_uncorrelated.npy', psf_med)


for i, img_file in enumerate(img_files[:]):

    # 1) Align, center and crop
    sci_img_processed, norm_const_1 = run_pre_processing(img_file = ori_folder + Path(img_file).stem,
                    img_key = 6667, 
                    dark_file = f'{ori_folder}darks_img',  
                    dark_key= 10000, 
                    ref_psf=f'{ori_folder}psf_', 
                    radius = radius,
                    method='masked_annulus',
                    normalisation= True
                    )

    # 2) Sort filename
    name = os.path.basename(img_file)          # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
    save_name = name.split("_coro")[0]  # int_1.0ro_10ws_no_coro

    if img_root_counts[save_name] > 1:
        img_counter[save_name] += 1
        filename = f"{save_name}_{img_counter[save_name]}.npy"
    else:
        filename = f"{save_name}.npy"

    # 3) Detect outliers
    sci_img_processed_no_outlier, outlier_matrix, norm_const_2 = run_outlier_detection(img_file = filename.split('.n')[0], 
                        sci_img_processed = sci_img_processed, 
                        n_components=3,
                        mad_threshold=6,
                        PCA_analysis = True,
                        PCA_plot_output= save_plot_path,
                        normalisation = True,
                        mask = mask_outlier
                        )

    # plot_outlier_images_by_pc(
    #     sci_img_processed, outlier_matrix
    #     )
    diff = np.mean(sci_img_processed, axis = 0) - np.mean(sci_img_processed_no_outlier, axis = 0)

    
    # 4) Save the file
   
    diffs_outliers_imgs[filename.split('.n')[0]] = (diff)
    raw_contrast_amplitudes[filename.split('.n')[0]] = norm_const_1 * norm_const_2 / norm_const_uncorelated * DIT_psf / DIT_img

    if create == True:
        np.save(save_path+filename, sci_img_processed_no_outlier)

    #for psf_file in psf_files[-2:]:
        # 5) Do the sme if psf corresponds
        # 5.1) Check if need to do psf

        psf_file = psf_files[i]
        psf_img_processed, _ = run_pre_processing(img_file = ori_folder + Path(psf_file).stem, 
                            img_key = 8475, 
                            dark_file = f'{ori_folder}darks_psf',  
                            dark_key= 10000, 
                            ref_psf=f'{ori_folder}psf_', 
                            radius = radius,
                            method='gaussian',
                            normalisation= True
                            )
        # 5.2) Detect outliers
        # psf_img_processed_no_outlier, outlier_matrix = run_outlier_detection(img_file = ori_folder + Path(psf_file).stem, 
        #                     sci_img_processed = psf_img_processed, 
        #                     n_components=3,
        #                     mad_threshold=6,
        #                     PCA_analysis = False,
        #                     PCA_plot_output= save_plot_path,
        #                     normalisation= True
        #                     )
        # plot_outlier_images_by_pc(
        #     psf_img_processed, outlier_matrix
        #     )
        # diff = np.mean(psf_img_processed, axis = 0) - np.mean(psf_img_processed_no_outlier, axis = 0)

        
        # 5.3) Save the file
        psf_med = np.mean(psf_img_processed, axis = 0)
        name = os.path.basename(psf_file)          # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0       
        save_name = name.split("_no")[0]  # int_1.0ro_10ws_no_coro

        if psf_root_counts[save_name] > 1:
            psf_counter[save_name] += 1
            filename = f"psf_{save_name}_{img_counter[save_name]}.npy"
        else:
            filename = f"psf_{save_name}.npy"


        # name = os.path.basename(psf_file)          # int_1.0ro_10ws_no_coro_2026-07-22T11-24-21.543_0
        # save_name = name.split("_no")[0]  # int_1.0ro_10ws_no_coro
        # np.save(f'{save_path}psf_{save_name}.npy', psf_med)
        
            np.save(save_path+filename, psf_med) 
        # diffs_outliers_psf[filename.split('.n')[0]] = (diff)

    plt.close('all')
    print(f'Done: {filename}')
    
# RUN AND SAVE ANALYSIS

out_path = '/home/aosimul/noah/results/animations/pre_processing/'
ori_path = '/home/aosimul/noah/data/ghost_images/10_20ws_2/'
int_ori_path = '/home/aosimul/noah/data/ghost_images/10_20ws/'

for ro in [0.5, 0.7, 1.0]: 
    for ws in [10, 20]:
            name = f"{ro}ro_{ws}ws" 

            int_cube = np.load( f"{int_ori_path}int_{name}.npy" ) 
            pred_cube = np.load( f"{int_ori_path}pred_{name}.npy" ) 

            # Speckle Linearity

            fig, fit_results = plot_symmetric_comparison( 
                int_cube, pred_cube, suptitle=f"Speckle symmetry: {ro} ro, {ws} ws", angle=[-45.0, 75.0, 15.0], dr = 3, r_aperture = 1) 
            plt.savefig(f'{out_path}linearity_{name}.png', dpi=400, bbox_inches="tight")
            #plt.show()

            # Outliers and Intensity Contrast


            fig = plt.figure(figsize = (15,6))

            # Main grid: left column narrow, right column wide
            gs = fig.add_gridspec(1, 2, width_ratios=[1,1], wspace=0.2)

            gs_left = gs[0].subgridspec(2, 2, wspace=0.45, hspace = 0.3)

            ax1 = fig.add_subplot(gs_left[0])
            ax2 = fig.add_subplot(gs_left[1])
            ax3 = fig.add_subplot(gs_left[2])
            ax4 = fig.add_subplot(gs_left[3])

            # Right image spans the entire right column
            ax_big = fig.add_subplot(gs[1])

            ax2.set_title('Integrator: Mean Pre-processed Frame and Outlier Deviation', loc = 'right')
            im = ax1.imshow(np.log10(np.mean(int_cube, axis = 0)))
            plt.colorbar(im, label='Log Normalised Counts', fraction =0.03, location = 'left', pad = 0.25)

            ax2.set_title(' ')
            im = ax2.imshow((diffs_outliers_imgs[f'int_{name}']))
            plt.colorbar(im, label='Normalised Counts', fraction =0.03, location = 'left', pad = 0.25)

            ax_big.set_title(' ')
            r_int, y_noncoro_int, _ = radial_profile_pre(np.mean(int_cube, axis = 0))
            r_pred, y_noncoro_pred, _ = radial_profile_pre(np.mean(pred_cube, axis = 0))
            ax_big.plot(r_int, y_noncoro_int * raw_contrast_amplitudes[f'int_{name}'], '--', color = 'k', lw = 2, label = 'Classic AO', alpha = 0.8)
            ax_big.plot(r_pred, y_noncoro_pred * raw_contrast_amplitudes[f'pred_{name}'], '-', color = 'k', lw = 2, label = 'Predictive AO', alpha = 0.8)
            ax_big.legend()
            ax_big.set_yscale('log')
            ax_big.set_xlim(left =0)
            #ax_big.set_ylim(3e-6, 1)
            ax_big.set_xlabel("Angular separation [FWHM]")
            ax_big.set_ylabel("Raw contrast")

            ax4.set_title('Predictive Control: Mean Pre-processed Frame and Outlier Deviation', loc = 'right')
            im = ax3.imshow(np.log10(np.mean(pred_cube, axis = 0)))
            plt.colorbar(im, label='Log Normalised Counts', fraction =0.03, location = 'left', pad = 0.25)

            ax4.set_title(' ')
            im = ax4.imshow((diffs_outliers_imgs[f'pred_{name}']))
            plt.colorbar(im, label='Normalised Counts', fraction =0.03, location = 'left', pad = 0.25)

            plt.suptitle(f'Atmospheric Condition: {ro} arcsec ro, {ws} m/s windspeed', fontweight = 'bold', fontsize = 16)
            
            plt.savefig(f'{out_path}raw_contrast_{name}.png', dpi=400, bbox_inches="tight")
            #plt.show()
            plt.close('all')

# save raw contrast coeef

with open(f'{save_path}RC_coef.pkl', 'wb') as f:
    pickle.dump(raw_contrast_amplitudes, f)