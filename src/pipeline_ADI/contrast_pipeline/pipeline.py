from pathlib import Path
import yaml
import numpy as np
import pandas as pd


import importlib
import src.pipeline_ADI.contrast_pipeline.functions_ADI as functions_ADI
importlib.reload(functions_ADI)
from src.pipeline_ADI.contrast_pipeline.functions_ADI import *

import applefy
importlib.reload(applefy)
from applefy.detections.contrast import Contrast
from applefy.utils import mag2flux_ratio
from applefy.utils.positions import center_subpixel


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(science_file,
                  dit_science,
                  dit_psf,
                  radius_psf,
                  radius_sc = None,
                  psf_file = None
                  ):

    sci_img = np.load(science_file)

    if radius_sc:
        sci_img = zoom_to_peak(sci_img, radius_sc)

    if psf_file:
        psf = np.load(psf_file)
    else:
        n_psf = round(dit_psf / dit_science)
        psf = np.sum(sci_img[:n_psf], axis=0)

    psf = zoom_to_peak(psf, radius_psf)


    return {
        "psf": psf,
        "sci_img": sci_img,
        "fwhm": calculate_fwhm(psf),
        "dit_psf": dit_psf,
        "dit_science": dit_science 
    }



def run_pipeline(config):

    root_dir = Path(".")

    inst = config["instrument"]
    ctr = config["contrast"]
    crv = config["curve"]

    datasets = {}

    for name, ds in config["datasets"].items():

        if not ds["enabled"]:
            continue

        datasets[name] = build_dataset(
            ds["science_file"],
            inst["dit_science"],
            inst["dit_psf"],
            inst["radius"]
        )

    algorithms = {
        k: k
        for k, enabled in config["algorithms"].items()
        if enabled
    }

    flux_ratio = mag2flux_ratio(
        ctr["flux_ratio_mag"]
    )


    for dataset_name, dataset in datasets.items():

        angles = np.linspace(
            ctr["angle_start"],
            ctr["angle_end"],
            dataset["sci_img"].shape[0]
        )

        angles = np.deg2rad(angles)

        for algo_name in algorithms:

            output_path = (
                root_dir /
                f"results/contrast_curves/"
                f"{config['experiment']['name']}"
                f"_{dataset_name}_{algo_name}"
            )

            output_path.mkdir(
                parents=True,
                exist_ok=True
            )

            contrast_instance = Contrast(
                science_sequence=dataset["sci_img"],
                psf_template=dataset["psf"],
                parang_rad=angles,
                psf_fwhm_radius=dataset["fwhm"] / 2,
                dit_psf_template=dataset["dit_psf"],
                dit_science=dataset["dit_science"],
                scaling_factor=ctr["scaling_factor"],
                checkpoint_dir=output_path
            )

            seps = np.arange(
                0,
                round(
                    center_subpixel(
                        dataset["sci_img"][0]
                    )[0]
                    * ctr["max_separation"]
                ),
                dataset["fwhm"] * ctr["separation"]
            )[1:]

            contrast_instance = fake_planet_experiment(
                contrast_instance,
                flux_ratio,
                ctr["num_fake_planets"],
                ctr["components"],
                version=algo_name,
                separations=seps,
                approx_svd=(
                    ctr["approx_svd_trunc"]
                    if ctr["approx_svd_trunc"]
                    else -1
                ),
                device=ctr["device"]
            )

            if config['curve']["enabled"]:
                
                curves = {}

                curves[(dataset_name, algo_name)] = (
                    compute_contrast_curves(
                        contrast_instance,
                        dataset["fwhm"],
                        pixel_scale=inst["pixel_size"],
                        photometry="AS",
                        test="t-test"
                    )
                )

                return curves


