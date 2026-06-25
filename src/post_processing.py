#from astropy.modeling import models, fitting
import pandas as pd
import importlib
import src.functions as functions
importlib.reload(functions)
from src.functions import *

import numpy as np
import os

#import seaborn as sns


from pathlib import Path
root_dir = Path(".")
print(root_dir)

import applefy
importlib.reload(applefy)

from applefy.detections.contrast import Contrast
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, \
    fpf_2_gaussian_sigma, LaplaceBootstrapTest

from applefy.utils.file_handling import load_adi_data
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.utils.positions import center_subpixel


frame_rate  = 1/400 #s
dit_science = 4e-3  #s
dit_psf     = 10    #s
pixel_size  = 0.005 #arcsec
LambdaD     = 4     #pixels
n_psf       = round(dit_psf/dit_science)
radius      = 20
sc_img_int  = np.load('/home/aosimul/noah/data/ghost_images/4ms/stg2_int.npy')
sc_img_pred = np.load('/home/aosimul/noah/data/ghost_images/4ms/stg2_pred.npy')
sc_img_int  = sc_img_int[:, ]
psf_int     = np.sum(sc_img_int[:n_psf],  axis = 0)
psf_int     = zoom_to_peak(psf_int, radius)
psf_pred    = np.sum(sc_img_pred[:n_psf], axis = 0)
psf_pred    = zoom_to_peak(psf_pred, radius)

sc_img_int     = sc_img_int.reshape(int(sc_img_int.shape[0]/5), 5, *sc_img_int.shape[1:]).sum(axis=1)
sc_img_pred    = sc_img_pred.reshape(int(sc_img_pred.shape[0]/5), 5, *sc_img_pred.shape[1:]).sum(axis=1)
dit_science    = dit_science*5

# CREATE DATASETS
datasets = {
    "int": {
        "psf"           : psf_int,
        "sci_img"       : sc_img_int,
        "fwhm"          : calculate_fwhm(psf_int),
        "dit_psf"       : dit_psf,          #s of integration time
        "dit_science"   : dit_science,     #s = 10 phase screens at 0.001s each
    },

    "pred": {
        "psf"           : psf_pred,
        "sci_img"       : sc_img_pred,
        "fwhm"          : calculate_fwhm(psf_pred),
        "dit_psf"       : dit_psf,          #s of integration time
        "dit_science"   : dit_science,  
    },
}

algorithms = {
    "PCAD": "PCAD",
    "CADI": "CADI",
}

curves = {}

#FILL
#-----------
# fake planet brightness
flux_ratio_mag          = 16
num_fake_planets        = 3
components              = [5, 10, 20, 50, 75, 100] 
scaling_factor          = 1.0  # A factor to account e.g. for ND filters
angles                  = np.linspace(0, 30, np.shape(sc_img_int)[0])   #parang[::10]
angles                  = np.deg2rad(angles)
flux_ratio              = mag2flux_ratio(flux_ratio_mag)
name                    = 'GHOST_no_coro52'
separation              = 1
max_separation          = 0.7                                           #in fraction of total image radius
approx_svd_trunc        = round(np.shape(sc_img_int)[0] / 3)
device                  = 'cuda'                                        #'cpu'
#-----------

for dataset_name, dataset in datasets.items():
    for algo_name, alg in algorithms.items():

        path = f"results/contrast_curves/{name}_{dataset_name}_{algo_name}"
        if not os.path.exists(path):
            os.makedirs(path)
            
        contrast_instance = Contrast(
            science_sequence    =dataset["sci_img"],
            psf_template        =dataset["psf"],
            parang_rad          =angles,
            psf_fwhm_radius     =dataset["fwhm"] / 2, # Diameter in pixel
            dit_psf_template    =dataset["dit_psf"], 
            dit_science         =dataset["dit_science"],  # integration time
            scaling_factor      =scaling_factor, # A factor to account e.g. for ND filters
            checkpoint_dir      =root_dir / Path(path)
            )
        seps =  None if separation == None else np.arange(0, round(center_subpixel(dataset["sci_img"][0])[0] * max_separation), dataset["fwhm"]* separation)[1:]
        contrast_instance = fake_planet_experiment(contrast_instance, flux_ratio, num_fake_planets, components, version = alg,
        separations = seps, 
        approx_svd = approx_svd_trunc if approx_svd_trunc else -1,
        device= device)
        curves[(dataset_name, algo_name)] = compute_contrast_curves(contrast_instance, dataset["fwhm"], pixel_scale=pixel_size,  photometry = 'AS', test = 't-test')

    # save the contrast curves in a single dataframe
    (curve, err) = curves[(dataset_name, "PCAD")]
    (curve_cadi, err_cadi) = curves[(dataset_name, "CADI")]

    curves[(dataset_name, "merged")] = (
        pd.merge(curve, curve_cadi, left_index= True, right_index=True, how="inner"),
        pd.merge(err, err_cadi, left_index= True, right_index=True, how="inner")
    )

