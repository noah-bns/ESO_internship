# from astropy.modeling import models, fitting

import importlib
import utils.functions as func
importlib.reload(func)
from utils.functions import *

import numpy as np
import os
import shutil
import pickle
from pathlib import Path

root_dir = Path(".")
print(root_dir)


from applefy_extensions.contrast_curves import ContrastFast
from fours.detection_limits.applefy_wrapper import (
    CADIDataReductionGPU,
    PCADataReductionGPU,
)
from applefy.utils import  mag2flux_ratio
from applefy.utils.positions import center_subpixel



frame_rate = 1 / 400  # s
dit_science = 4e-3  # s
dit_psf = 10  # s
pixel_size = None  # arcsec
# LambdaD     = 4     #pixels
n_psf = round(dit_psf / dit_science)

radius_psf = 10
radius_sc = 50

sc_img_int = np.load("/home/aosimul/noah/data/ghost_images/4ms/stg2_int.npy")
sc_img_pred = np.load("/home/aosimul/noah/data/ghost_images/4ms/stg2_pred.npy")
psf_pred = np.load("/home/aosimul/noah/data/ghost_images/4ms/psf_pred.npy")
psf_int = np.load("/home/aosimul/noah/data/ghost_images/4ms/psf_int.npy")
# sc_img_int  = sc_img_int[:, ]
# psf_int     = np.sum(sc_img_int[:n_psf],  axis = 0)
# psf_int     = zoom_to_peak(psf_int, radius_psf)
# psf_pred    = np.sum(sc_img_pred[:n_psf], axis = 0)
# psf_pred    = zoom_to_peak(psf_pred, radius_psf)

# sc_img_int = sc_img_int[:, radius_sc:-radius_sc, radius_sc:-radius_sc]
# sc_img_pred = sc_img_pred[:, radius_sc:-radius_sc, radius_sc:-radius_sc]
# psf_pred       = psf_pred[radius_psf:-radius_psf, radius_psf:-radius_psf]
# psf_int        = psf_int[radius_psf:-radius_psf, radius_psf:-radius_psf]

binning        = 25
sc_img_int     = sc_img_int.reshape(int(sc_img_int.shape[0]/binning), binning, *sc_img_int.shape[1:]).sum(axis=1)
sc_img_pred    = sc_img_pred.reshape(int(sc_img_pred.shape[0]/binning), binning, *sc_img_pred.shape[1:]).sum(axis=1)
dit_science    = dit_science*binning

# CREATE DATASETS
datasets = {
    "int": {
        "psf": psf_int,
        "sci_img": sc_img_int,
        "fwhm": calculate_fwhm(psf_int),
        "dit_psf": dit_psf,  # s of integration time
        "dit_science": dit_science,  # s = 10 phase screens at 0.001s each
    },
    "pred": {
        "psf": psf_pred,
        "sci_img": sc_img_pred,
        "fwhm": calculate_fwhm(psf_pred),
        "dit_psf": dit_psf,  # s of integration time
        "dit_science": dit_science,
    },
}

algorithms = {
    "PCAD": "PCAD",
    "CADI": "CADI",
}

curves = {}

# FILL
# -----------
contrast_result_dir = "/home/aosimul/noah/results/contrast_grid"
name = "test_grid_nobin"
grid = True
flux_ratio_mag = 16
flux_ratios_mag = np.linspace(4, 17, 5)
num_fake_planets = 2
components = [5, 20, 50, 100, 150]
scaling_factor = 1.0  # A factor to account e.g. for ND filters
angles = np.linspace(0, 30, np.shape(sc_img_int)[0])  # parang[::10]
angles = np.deg2rad(angles)
flux_ratio = mag2flux_ratio(flux_ratio_mag)
flux_ratios = mag2flux_ratio(flux_ratios_mag)
separation = 1
max_separation = 1  # in fraction of total image radius
approx_svd_trunc = round(np.shape(sc_img_int)[0] / 5)
device = "cpu"  #'cpu'
# -----------

for dataset_name, dataset in datasets.items():

    if grid == True:
        path = f"{contrast_result_dir}/{name}_{dataset_name}" #_{algo_name}"
    else:
        path = f"/home/aosimul/noah/results/contrast_curves/{name}_{dataset_name}_{algo_name}"

    if os.path.exists(path):
        shutil.rmtree(Path(path))
        print(f"Removed existing directory to avoid overwrites: {path}.")

    if not os.path.exists(path):
        os.makedirs(path)

    contrast_instance = ContrastFast(
        science_sequence=dataset["sci_img"],
        psf_template=dataset["psf"],
        parang_rad=angles,
        psf_fwhm_radius=dataset["fwhm"] / 2,  # Diameter in pixel
        dit_psf_template=dataset["dit_psf"],
        dit_science=dataset["dit_science"],  # integration time
        device=device,
        scaling_factor=scaling_factor,  # A factor to account e.g. for ND filters
        checkpoint_dir=root_dir / Path(path),
    )
    seps = (
        None
        if separation == None
        else np.arange(
            0,
            round(center_subpixel(dataset["sci_img"][0])[0] * max_separation),
            dataset["fwhm"] * separation,
        )[1:]
    )
    # contrast_instance = fake_planet_experiment(contrast_instance, flux_ratios, num_fake_planets, components, version = alg,
    #     separations = seps,
    #     #approx_svd = approx_svd_trunc if approx_svd_trunc else -1,
    #     device= device
    #     )
    # curves[(dataset_name, algo_name)] = compute_contrast_curves(contrast_instance, dataset["fwhm"], pixel_scale=pixel_size,  photometry = 'AS', test = 't-test', grid = grid)

    # # save the contrast curves in a single dataframe
    # (curve, err) = curves[(dataset_name, "PCAD")]
    # (curve_cadi, err_cadi) = curves[(dataset_name, "CADI")]

    # curves[(dataset_name, "merged")] = (
    #     pd.merge(curve, curve_cadi, left_index= True, right_index=True, how="inner"),
    #     pd.merge(err, err_cadi, left_index= True, right_index=True, how="inner")
    # )

    contrast_instance.design_fake_planet_experiments(
        flux_ratios=flux_ratios,
        num_planets=num_fake_planets,
        separations=seps,
        overwrite=True,
    )

    # 5.) Run cADI ------------------------------------------------------------
    cadi_algorithm_function = CADIDataReductionGPU(0)
    contrast_instance.run_fake_planet_experiments(
        algorithm_function=cadi_algorithm_function, num_parallel=1
    )

    old_results = contrast_instance.results_dict

    work_dir = contrast_instance.scratch_dir / Path("tensorboard_pca")
    work_dir.mkdir(exist_ok=True)

    pca_algorithm_function = PCADataReductionGPU(
        approx_svd=max(components)*10,  # needed due to limited GPU memory
        pca_numbers=np.array(components),
        device=0,
        work_dir=work_dir,
        special_name="try_pipeline",
        verbose=False,
    )

    contrast_instance.run_fake_planet_experiments(
        algorithm_function=pca_algorithm_function, num_parallel=1
    )


    contrast_instance.results_dict.update(old_results)

    # save the contrast instance as pickle
    print("Saving the contrast instance to disk ...", end=" ")
    contrast_instance_file = contrast_result_dir / Path(
        dataset_name + "_contrast_instance.pkl"
    )

    with open(contrast_instance_file, "wb") as f:
        pickle.dump(contrast_instance, f)

contrast_grids = compute_contrast_curves(
    contrast_instance,
    dataset["fwhm"],
    pixel_scale=pixel_size,
    photometry="AS",
    test="t-test",
    grid=grid,
)

# save the contrast grids
print("Saving the contrast grids to disk ...", end=" ")
contrast_grids_file = contrast_result_dir / Path(
    dataset_name + "_contrast_grids.pkl"
)

with open(contrast_grids_file, "wb") as f:
    pickle.dump(contrast_grids, f)

print("[DONE]")
