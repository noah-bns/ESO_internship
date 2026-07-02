from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import shutil

import importlib
from . import functions_ADI
importlib.reload(functions_ADI)
from .functions_ADI import *

import applefy
importlib.reload(applefy)
from applefy.detections.contrast import Contrast
from applefy.utils import mag2flux_ratio
from applefy.utils.positions import center_subpixel


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
        'experiment': ['name'],
        'datasets': None,  # At least one dataset must be enabled
        'instrument': ['dit_science', 'dit_psf'],
        'fake_planet': ['flux_ratio_mag', 'num_fake_planets', 'components'],
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
                  frame_rate, 
                  radius_sc,
                  psf_file
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
    
    Returns
    -------
    dict
        Dataset dictionary containing psf, sci_img, fwhm, dit_psf, dit_science.
    """

    sci_img = np.load(science_file)

    if radius_sc is not None:
        sci_img = zoom_to_peak(sci_img, radius_sc)

    if psf_file is not None:
        psf = np.load(psf_file)
    else:
        if frame_rate is None:
            raise ValueError("frame_rate must be provided if psf_file is None.")
        n_psf = round(dit_psf / frame_rate)
        psf = np.sum(sci_img[:n_psf], axis=0)

    psf = zoom_to_peak(psf, radius_psf)


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
    crv = config["curves"]
    

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
            frame_rate=inst["frame_rate"]
        
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
            # if crv["enabled"]:
                
            #     print(f"Computing contrast curves for {dataset_name} - {algo_name}...")
                
            #     curves_output_path = (
            #         root_dir /
            #         Path(f"{crv['path']}"
            #         f"/{config['experiment']['name']}"
            #         f"_{dataset_name}_{algo_name}"
            #         )
            #     )
                
            #     curves_output_path.mkdir(
            #         parents=True,
            #         exist_ok=True
            #     )

            #     curves = compute_contrast_curves(
            #         contrast_instance,
            #         dataset["fwhm"],
            #         pixel_scale=inst["pixel_size"],
            #         photometry=crv["photometry"],
            #         test=crv["test"]
            #     )

            #     all_curves[(dataset_name, algo_name)] = curves
                
            #     # Save results
            #     if crv["save_csv"] is True:
            #         _save_curves_csv(curves, curves_output_path, dataset_name, algo_name)
                
            #     if crv["save_plots"] is True:
            #         _save_curves_plot(curves, curves_output_path, dataset_name, algo_name)

    return all_curves if all_curves else None


def _save_curves_csv(curves, output_path, dataset_name, algo_name):
    """
    Save contrast curves to CSV file.
    
    Parameters
    ----------
    curves : dict or pd.DataFrame
        Contrast curve data.
    output_path : Path
        Output directory path.
    dataset_name : str
        Dataset name for filename.
    algo_name : str
        Algorithm name for filename.
    """
    csv_path = output_path / f"contrast_curve_{dataset_name}_{algo_name}.csv"
    
    if isinstance(curves, pd.DataFrame):
        curves.to_csv(csv_path, index=False)
    else:
        # Adapt based on your curve data structure
        pd.DataFrame(curves).to_csv(csv_path, index=False)
    
    print(f"Saved CSV: {csv_path}")


def _save_curves_plot(curves, output_path, dataset_name, algo_name):
    """
    Save contrast curves as plot.
    
    Parameters
    ----------
    curves : dict or pd.DataFrame
        Contrast curve data.
    output_path : Path
        Output directory path.
    dataset_name : str
        Dataset name for filename.
    algo_name : str
        Algorithm name for filename.
    """
    import matplotlib.pyplot as plt
    
    plot_path = output_path / f"contrast_curve_{dataset_name}_{algo_name}.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Adapt based on your curve data structure
    if isinstance(curves, pd.DataFrame):
        ax.plot(curves.iloc[:, 0], curves.iloc[:, 1], 'o-', linewidth=2)
    else:
        ax.plot(curves, 'o-', linewidth=2)
    
    ax.set_xlabel("Separation (pixels)")
    ax.set_ylabel("Contrast")
    ax.set_title(f"Contrast Curve: {dataset_name} - {algo_name}")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    
    # Example usage
    config_file = "configs/ghost.yaml"
    defaults_file = "configs/default_values.yaml"
    
    # try:
    #     curves = run_pipeline(config_file, defaults_file)
        
    #     if curves:
    #         print(f"\n✓ Pipeline completed successfully!")
    #         print(f"  Generated {len(curves)} contrast curve(s)")
    #     else:
    #         print("\n✓ Pipeline completed. No curves computed (disabled in config).")
            
    # except FileNotFoundError as e:
    #     print(f"✗ Error: Configuration file not found: {e}")
    # except ValueError as e:
    #     print(f"✗ Configuration error: {e}")
    # except Exception as e:
    #     print(f"✗ Pipeline error: {e}")
    #     raise
    config = load_config(config_file, defaults_file)
    print(config)
