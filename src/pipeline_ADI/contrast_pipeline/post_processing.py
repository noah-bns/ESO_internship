from pathlib import Path
import yaml
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from copy import deepcopy

import importlib
from . import functions_ADI
importlib.reload(functions_ADI)
from .functions_ADI import *


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


def _load_angles(angle_file, size):
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
        ang = np.load(angle_file)
        return ang[:size]
    elif angle_file.suffix == '.csv':
        ang = np.loadtxt(angle_file, delimiter=',')
        return ang[:size]
    else:
        raise ValueError(f"Unsupported angle file format: {angle_file.suffix}, must be '.npy' or '.csv'")
    

def run_pipeline_old(config):
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

    inst    = config["instrument"]
    fp      = config["fake_planet"]
    cnst    = config["contrast"]
    exp     = config['experiment']['name']

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
            angles = _load_angles(fp["angle_file"], np.shape(dataset["sci_img"])[0])
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
                f"{exp}"
                f"_{dataset_name}_{algo_name}"
            ))

            grid = True if isinstance(fp['flux_ratio_mag'], (list, np.ndarray)) else False

            contrast_instance = fake_planet_experiment(
                    contrast_instance =contrast_instance,
                    output_path = output_path,
                    dataset = dataset,
                    fp_config = fp,
                    separations = seps,
                    algo_name = algo_name,
                    angles = angles,
                    )
            
            # Compute contrast curves if enabled
            if cnst["enabled"]:
                
                curves_output_path = (
                    root_dir /
                    Path(f"{cnst['path']}"
                    f"/{exp}"
                    )
                )
                
                curves_output_path.mkdir(
                    parents=True,
                    exist_ok=True
                )

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
                    print(f"Computing contrast grid for {dataset_name} - {algo_name}...")
                
                    _save_grid_animation(dataset_contrast[algo_name][1], curves_output_path, exp+'_'+dataset_name)
        

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

    inst    = config["instrument"]
    fp      = config["fake_planet"]
    cnst    = config["contrast"]
    exp     = config['experiment']['name']

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
            angles = _load_angles(fp["angle_file"], np.shape(dataset["sci_img"])[0])
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
        

        first_algo = next(iter(algorithms))

        # Set output directory

        print(f"\nProcessing {dataset_name} with {first_algo}...")

        output_path = (
            root_dir /
            Path(
            f"{fp["path"]}/"
            f"{exp}"
            f"_{dataset_name}"
        ))

        # Remove existing directory and all its contents
        if output_path.exists():
            shutil.rmtree(output_path)
            print(f"Removed existing directory to avoid overwrites: {output_path}.")

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        flux_ratio = fp['flux_ratio_mag']
        if isinstance(flux_ratio, list):
            flux_ratio = np.array(flux_ratio)
        flux_ratio = mag2flux_ratio(flux_ratio)


        grid = True if isinstance(fp['flux_ratio_mag'], (list, np.ndarray)) else False


        contrast_instance = ContrastFast(
            science_sequence=dataset["sci_img"],
            psf_template=dataset["psf"],
            parang_rad=angles,
            psf_fwhm_radius=dataset["fwhm"] / 2,
            dit_psf_template=dataset["dit_psf"],
            device = fp['device'],
            dit_science=dataset["dit_science"],
            scaling_factor=fp["scaling_factor"],
            checkpoint_dir= output_path
        )

        contrast_instance.design_fake_planet_experiments(
            flux_ratios= flux_ratio,
            num_planets=fp['num_fake_planets'],
            separations = seps,
            overwrite=True,
            )


        contrast_instance = fake_planet_experiment(
                contrast_instance = contrast_instance,
                output_path = output_path,
                dataset = dataset,
                fp_config = fp,
                separations = seps,
                algo_name = first_algo,
                angles = angles,
                )

        for algo_name, alg in list(algorithms.items())[1:]:

            old_results = contrast_instance.results_dict

            contrast_instance = fake_planet_experiment(
                    contrast_instance = contrast_instance,
                    output_path = output_path,
                    dataset = dataset,
                    fp_config = fp,
                    separations = seps,
                    algo_name = algo_name,
                    angles = angles,
                    )
            
            contrast_instance.results_dict.update(old_results)

        # Compute contrast curves if enabled
        if cnst["enabled"]:
            
            curves_output_path = (
                root_dir /
                Path(f"{cnst['path']}"
                f"/{exp}"
                )
            )
            
            curves_output_path.mkdir(
                parents=True,
                exist_ok=True
            )

            contrasts_output = compute_contrast(
                contrast_instance,
                dataset["fwhm"],
                pixel_scale=inst["pixel_size"],
                photometry=cnst["photometry"],
                test=cnst["test"],
                grid = grid
            )
    
            # Save grid results
            if grid ==True:
                print(f"Computing the Overall best grid for {dataset_name} ...")
                baseline_grids = deepcopy(contrasts_output[1])
                del baseline_grids["cADI"]

                all_grids = np.array(
                    [fpf_2_gaussian_sigma(tmp_grid.values)
                    for tmp_grid in baseline_grids.values()])

                # Best values
                best_values = np.max(all_grids, axis=0)

                # Which grid produced the best value
                best_idx = np.argmax(all_grids, axis=0)

                # Map indices to PCA numbers
                pca_numbers = np.array([
                    int(key.split("_")[2])
                    for key in baseline_grids.keys()
                ])

                best_pca = pca_numbers[best_idx]

                # Store as DataFrames
                best_value_df = deepcopy(next(iter(baseline_grids.values())))
                best_value_df.iloc[:, :] = best_values
                best_value_df.index = flux_ratio2mag(best_value_df.index)

                best_pca_df = deepcopy(best_value_df)
                best_pca_df.iloc[:, :] = best_pca
                best_pca_df.index = flux_ratio2mag(best_pca_df.index)

                print(f"Computing contrast grid for {dataset_name} ...")
            
                _save_grid_animation(contrasts_output[1], curves_output_path, exp+'_'+dataset_name)
                plot_overall_best(best_value_df, curves_output_path, dataset_name)
                plot_overall_best(best_pca_df, curves_output_path, dataset_name, pca = True)
                rangey = (5,15)
                result = plot_contrast_curves(contrasts_output[0], rangey, curves_output_path, exp+'_'+dataset_name, cmap = 'winter', title =(r"$5 \sigma_{\mathcal{N}}$ Contrast Curves" +f"\n{exp} -- {dataset_name}"))

            else:
                result = plot_contrast_curves(contrasts_output[0], rangey, curves_output_path, exp+'_'+dataset_name, contrast_errors = contrasts_output[1], cmap = 'winter', title =(r"$5 \sigma_{\mathcal{N}}$ Contrast Curves" +f"\n{exp} -- {dataset_name}"))

            if cnst['save_csv']:
                result.to_hdf(
                    f"{curves_output_path}/overall_best.h5",
                    key=exp+'_'+dataset_name,
                    mode="a"
                )


def _save_grid_animation(contrast_grids, curves_output_path, dataset_name):

    keys = list(contrast_grids.keys())

    fig = plt.figure(figsize=(7, 4))

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

        algo_name = 'CADI' if key =='cADI' else 'PCAD'

        plot_contrast_grid(
            contrast_grid_axis=contrast_ax,
            colorbar_axis=colorbar_ax,
            contrast_grid=grid,
            cmap = 'YlOrRd' if algo_name == 'CADI' else "YlGnBu"
        )

        contrast_ax.set_ylabel("Contrast - $c=f_p/f_*$ [mag]", fontsize=14)
        contrast_ax.set_xlabel("Separation [FWHM]", fontsize=14)

        contrast_ax.set_title(
            f"{dataset_name.replace("_", " ")}: {key.replace("_", " ")}",
            fontsize=16,
            fontweight="bold"
        )

        contrast_ax.tick_params(labelsize=12)
        plt.subplots_adjust(bottom=0.2)
        plt.close()

    ani = FuncAnimation(
        fig,
        update,
        frames=len(keys),
        interval=500
    )

    ani.save(f"{(curves_output_path)}/GRID_{dataset_name}.gif", writer=PillowWriter(fps=1))
    