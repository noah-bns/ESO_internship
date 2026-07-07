from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import shutil
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

import importlib
from . import functions_ADI
importlib.reload(functions_ADI)
from .functions_ADI import *

import applefy
importlib.reload(applefy)
from applefy.detections.contrast import Contrast
from applefy.utils import mag2flux_ratio
from applefy.utils.positions import center_subpixel
from applefy.statistics import fpf_2_gaussian_sigma

def load_config(config_path, defaults_path="configs/default_values.yaml"):
    """
    Load experiment config and merge with default values.
    
    Parameters
    ----------
    config_path : str or Path
        Path to experiment configuration file.
    defaults_path : str or Path
        Path to default values configuration file.
    
    Returns
    -------
    dict
        Merged configuration with all values populated.
    """
    # Load defaults
    with open(defaults_path, "r") as f:
        defaults = yaml.safe_load(f)
    
    # Load experiment config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Merge configs: experiment values override defaults
    merged_config = deep_merge_dicts(defaults, config)
    
    # Validate required fields
    _validate_required_fields(merged_config)
    
    return merged_config


def deep_merge_dicts(defaults, overrides):
    """
    Recursively merge overrides into defaults.
    
    Parameters
    ----------
    defaults : dict
        Default configuration values.
    overrides : dict
        User-provided configuration values (overrides defaults).
    
    Returns
    -------
    dict
        Merged configuration.
    """
    merged = defaults.copy()
    
    if overrides is None:
        return merged
    
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def _validate_required_fields(config):
    """
    Validate that all required fields are present.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to validate.
    
    Raises
    ------
    ValueError
        If required fields are missing.
    """
    required_fields = {
        'experiment': ['name'],    # Will default to 'test'
        'datasets': None,
        'instrument': ['dit_science', 'dit_psf'],
        'algorithms': None,
        # Optional fields with enforced defaults:
        'fake_planet': [
            'flux_ratio_mag',           # Will default to 16
            'num_fake_planets',         # Will default to 3
            'components',               # Will default to [10, 20, 50, 75, 100]
            'separation',               # Will default to 1.0
            'max_separation',           # Will default to 1.0
            'path',                     # Will default to "results/fake_planets"
        ],
    }
    
    # Check top-level required fields
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required section: '{section}'")
        
        if fields is None:  # Special handling for datasets
            if 'datasets' in config:
                enabled_datasets = [
                    name for name, ds in config['datasets'].items()
                    if ds.get('enabled', False)
                ]
                if not enabled_datasets:
                    raise ValueError("At least one dataset must be enabled.")
        else:
            for field in fields:
                if field not in config[section] or config[section][field] is None:
                    raise ValueError(
                        f"Missing required field: '{section}.{field}'"
                    )


def build_dataset(science_file,
                  dit_science,
                  dit_psf,
                  radius_psf,
                  radius_sc,
                  psf_file,
                  dit_factor,
                  ):
    """
    Build dataset dictionary from input files.
    
    Parameters
    ----------
    science_file : str or Path
        Path to science cube.
    dit_science : float
        Science frame integration time (seconds).
    dit_psf : float
        PSF calibration integration time (seconds).
    radius_psf : int
        Radius for PSF extraction (pixels).
    radius_sc : int, optional
        Radius for science image extraction (pixels).
    psf_file : str or Path, optional
        Path to external PSF file. If None, PSF is generated from science cube.
    dit_factor : int, optional
        Factor to multiply the integration time (saves computation time by combining science frames (e.g., 2 means sum every 2 frames).

            Returns
    -------
    dict
        Dataset dictionary containing psf, sci_img, fwhm, dit_psf, dit_science.
    """

    sci_img = np.load(science_file)

    if radius_sc is not None:
        sci_img = zoom_to_peak(sci_img, radius_sc)
    else:
        print("Science images assumed square and centered")

    if psf_file is not None:
        psf = np.load(psf_file)
    else:
        n_psf = round(dit_psf / dit_science)
        psf = np.sum(sci_img[:n_psf], axis=0)
    
    if radius_psf is not None:
        psf = zoom_to_peak(psf, radius_psf)
    else:
        print("PSF assumed square and centered. No zooming applied.")

    if dit_factor is not None:
        sci_img      = sci_img.reshape(int(sci_img.shape[0]/dit_factor), dit_factor, *sci_img.shape[1:]).sum(axis=1)
        dit_science *= dit_factor
        print(f"Science DIT multiplied by factor {dit_factor}. New DIT: {dit_science}s")

    return {
        "psf": psf,
        "sci_img": sci_img,
        "fwhm": calculate_fwhm(psf),
        "dit_psf": dit_psf,
        "dit_science": dit_science 
    }


def _load_angles(angle_file):
    """
    Load parallactic angles from file.
    
    Parameters
    ----------
    angle_file : str or Path
        Path to angle file (.npy, .csv, etc.).
    
    Returns
    -------
    np.ndarray
        Parallactic angles in degrees.
    """
    angle_file = Path(angle_file)
    
    if angle_file.suffix == '.npy':
        return np.load(angle_file)
    elif angle_file.suffix == '.csv':
        return np.loadtxt(angle_file, delimiter=',')
    else:
        raise ValueError(f"Unsupported angle file format: {angle_file.suffix}, must be '.npy' or '.csv'")
    

def run_pipeline(config):
    """
    Main pipeline execution function.
    
    Reads configuration, builds datasets, and runs all enabled algorithms
    to compute contrast curves.
    
    Parameters
    ----------
    config : dict
        Experiment configuration dictionary.
    
    Returns
    -------
    dict or None
        Contrast curves if enabled, otherwise None.
    """

    root_dir = Path(".")

    inst = config["instrument"]
    fp = config["fake_planet"]
    cnst = config["contrast"]
    

    algorithms = {
        k: k
        for k, enabled in config["algorithms"].items()
        if enabled
    }

    if not algorithms:
        raise ValueError("No algorithms enabled in configuration.")


    datasets = {}

    for name, ds in config["datasets"].items():

        if not ds.get("enabled", False):
            continue

        print(f"Building dataset: {name}")

        datasets[name] = build_dataset(
            science_file=ds["science_file"],
            dit_science=inst["dit_science"],
            dit_psf=inst["dit_psf"],
            radius_psf=inst["radius_psf"],
            radius_sc=inst.get("radius_sc"),
            psf_file=ds.get("psf_file"),
            dit_factor=inst["dit_factor"]
        )

    if not datasets:
        raise ValueError("No datasets enabled in configuration.")
    

    # Store all contrast curves
    all_curves = {}

    # Process each dataset with each algorithm
    for dataset_name, dataset in datasets.items():

        # Generate parallactic angles
        if "angle_file" in fp and fp["angle_file"] is not None:
            # Load angles from file
            print(f"Loading angles from file (in degrees): {fp['angle_file']}")
            angles = _load_angles(fp["angle_file"])
        else:
            # Generate angles
            print(f"Generating angles from {fp['angle_start']} to {fp['angle_end']} degrees.")
            angles = np.linspace(
                fp["angle_start"],
                fp["angle_end"],
                dataset["sci_img"].shape[0]
            )

        angles = np.deg2rad(angles)

        # Calculate separation range
        center_coords = center_subpixel(dataset["sci_img"][0])
        max_sep_pixels = round(center_coords[0] * fp["max_separation"])
        
        seps = np.arange(
            0,
            max_sep_pixels,
            dataset["fwhm"] * fp["separation"]
        )[1:]
    
        # Run fake planet experiment
        print(f"Running fake planet experiment with {fp['num_fake_planets']} planets and components {fp['components']}...")
        
        # Store contrast curves per dataset
        dataset_contrast = {}   

        for algo_name in algorithms:

            print(f"\nProcessing {dataset_name} with {algo_name}...")

            output_path = (
                root_dir /
                Path(
                f"{fp["path"]}/"
                f"{config['experiment']['name']}"
                f"_{dataset_name}_{algo_name}"
            ))

            contrast_instance = fake_planet_experiment(
                    output_path = output_path,
                    dataset = dataset,
                    fp_config = fp,
                    separations = seps,
                    algo_name = algo_name,
                    angles = angles
                    )
            
            # Compute contrast curves if enabled
            if cnst["enabled"]:
                
                print(f"Computing contrast curves for {dataset_name} - {algo_name}...")
                
                curves_output_path = (
                    root_dir /
                    Path(f"{cnst['path']}"
                    f"/{config['experiment']['name']}"
                    )
                )
                
                curves_output_path.mkdir(
                    parents=True,
                    exist_ok=True
                )

                grid = True if isinstance(fp['flux_ratio_mag'], (list, np.ndarray)) else False

                dataset_contrast[algo_name] = compute_contrast(
                    contrast_instance,
                    dataset["fwhm"],
                    pixel_scale=inst["pixel_size"],
                    photometry=cnst["photometry"],
                    test=cnst["test"],
                    grid = grid
                )
        
                # Save grid results
                if grid ==True:
                    _save_grid_animation(dataset_contrast[algo_name][1], curves_output_path, dataset_name, algo_name)



def _save_grid_animation(contrast_grids, curves_output_path, dataset_name, algo_name):

    keys = list(contrast_grids.keys())

    fig = plt.figure(figsize=(8, 4))

    gs0 = fig.add_gridspec(1, 1)
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs0[0],
        wspace=0.05,
        width_ratios=[1, 0.03]
    )

    contrast_ax = fig.add_subplot(gs1[0])
    colorbar_ax = fig.add_subplot(gs1[1])


    def update(i):
        contrast_ax.clear()
        colorbar_ax.clear()

        key = keys[i]

        grid = contrast_grids[key].copy()

        # convert FPF to sigma
        grid = grid.map(fpf_2_gaussian_sigma)

        # convert flux ratio to magnitude
        grid.index = flux_ratio2mag(grid.index)

        plot_contrast_grid(
            contrast_grid_axis=contrast_ax,
            colorbar_axis=colorbar_ax,
            contrast_grid=grid,
            cmap = 'YlOrRd' if algo_name == 'CADI' else "YlGnBu"
        )

        contrast_ax.set_ylabel("Contrast - $c=f_p/f_*$ [mag]", fontsize=14)
        contrast_ax.set_xlabel("Separation [FWHM]", fontsize=14)

        contrast_ax.set_title(
            f"{dataset_name} {algo_name}: {key.replace("_", " ")}",
            fontsize=16,
            fontweight="bold"
        )

        contrast_ax.tick_params(labelsize=12)


    ani = FuncAnimation(
        fig,
        update,
        frames=len(keys),
        interval=500
    )

    ani.save(f"{curves_output_path}/GRID.gif", writer=PillowWriter(fps=2))