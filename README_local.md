# README_local — what is where in `noah/`

**Scope:** this file only explains the folder layout and what each file is for, so people on the bench can find things. For the science context, the pipeline stages, installation and the "things to know" (hard-coded paths, acquisition window, etc.) read [README.md](README.md).

Legend: ✅ tracked in git · 📦 local only (gitignored: data, results, environments, forks) · 🕰 legacy (predates the `src/` layout, kept for reference, imports may need fixing).

```text
noah/
├── README.md             ✅ project README (context, pipelines, install)
├── README_local.md       ✅ this file
├── pyproject.toml        ✅ makes src/ installable as the `aosimul-pipelines` package (pip install -e .)
├── requirements.txt      ✅ pinned environment (Python 3.12, torch + CUDA)
├── .gitignore            ✅ ignores data/, results/, env_hci/, applefy_fixed/, media files
├── .vscode/settings.json 📦 hides env/data/results folders from the VS Code explorer and search
├── config/               ✅ PynPoint_config.ini, only for the PynPoint-based legacy notebooks
├── src/                  ✅ the two pipelines + shared utils (see below)
├── notebooks/            ✅ analysis notebooks and scripts (+ 📦 notebooks/results/)
├── data/                 📦 bench data and processed stacks (see below)
├── applefy_fixed/        📦 patched fork of applefy, needed by pipeline_ADI (pip install -e ./applefy_fixed)
├── env_hci/, .venv/      📦 virtual environments (use env_hci)
└── __pycache__/          📦 python cache, ignore
```

---

## `src/` — installed package

### `src/pipeline_pre_ADI/` — stage 1: raw `.tif` → cleaned `.npy` stacks

| File | One line |
|---|---|
| [README.md](src/pipeline_pre_ADI/README.md) | Detailed doc of the pre-processing functions, flowcharts and parameters |
| `pre_process/pre_processing.py` | Core routine: `run_pre_processing` (align, centre, dark-subtract, crop, normalise) and `run_outlier_detection` (PCA bad-frame rejection) |
| `pre_process/centering.py` | Centre finding: satellite-speckle detection, 2D Gaussian fit, coronagraph-mask circle fit |
| `pre_process/correlation.py` | Spatial symmetry and temporal correlation measures on stacks |
| `pre_process/plot_pre_processing.py` | Plots for the pre-processing: outlier frames per PC, raw contrast, symmetric comparisons |
| `notebooks/run.py` | **Entry point.** Runs the whole stage 1 on a folder of `.tif` files driven by a YAML config (`python run.py --config config_pre_process.yaml`) |
| `notebooks/config_pre_process.yaml` | Config for `run.py`: input folder, output roots, run name, modes |
| `notebooks/run_pre_processing.py` | 🕰 Older scripted version of `run.py` with hard-coded paths |
| `notebooks/analysis_pre_process.py` | Analysis after stage 1: raw contrast and correlation on the processed stacks |
| `notebooks/make_plots.ipynb` | Visualises the stage 1 outputs (outlier diffs, radial profiles) from `results/<run>/*.h5` |
| `results/` | 📦 stage 1 outputs per run (`10_15_20ws`, `bench`, `long_7ro_20ws`, ...): plots, json, h5 |

### `src/pipeline_ADI/` — stage 2: fake planets, PSF subtraction, contrast curves

| File | One line |
|---|---|
| [README.md](src/pipeline_ADI/README.md) | Detailed doc of the contrast pipeline, config keys and flowcharts |
| `contrast_pipeline/post_processing.py` | **Main module.** `load_config`, `build_dataset`, `run_pipeline`, `run_from_checkpoint` |
| `contrast_pipeline/functions_ADI.py` | Fake-planet experiment, contrast computation, parallactic angles, zoom helpers |
| `contrast_pipeline/cadi_utils.py` | Classical ADI (median/mean subtraction) on CPU and GPU, applefy wrapper classes |
| `contrast_pipeline/pca_utils.py` | PCA PSF subtraction on GPU (torch), several SVD methods, applefy wrapper |
| `contrast_pipeline/raw_contrast_utils.py` | "No subtraction" reduction, gives the raw contrast baseline |
| `contrast_pipeline/plotting.py` | Contrast curves, contrast grids, grid animations, overall-best plots |
| `contrast_pipeline/cli.py` | Command line entry: `-c experiment.yaml` → `run_pipeline` |
| `config_files/default_values.yaml` | Default values merged under every experiment config |
| `config_files/template.yaml` | Commented template to copy for a new experiment |
| `config_files/ghost.yaml`, `ghost_ref.yaml` | Experiment configs used on the GHOST data (current and reference) |
| `run.py` | Scripted parameter sweeps: loops over runs/settings and calls `run_pipeline` |
| `results/` | 📦 stage 2 outputs: `contrast/<experiment>/` (curves, grids, `overall_best.h5`) and `fake_planets/<experiment>/` (residual cubes, very large) |

### `src/utils/` — shared helpers

| File | One line |
|---|---|
| `naming.py` | Run tags (`7ro_20ws_1key_1-4`), name parsing for titles, loaders for stacks, raw `.tif` and contrast tables |
| `statistics.py` | FWHM, aperture masks, photometry, optimal SVD rank, correlation and mutual-information measures |

---

## `notebooks/` — analysis and exploration

### Current (import from the installed `src` packages)

| File | One line |
|---|---|
| `temporal_decorrelation_clean.ipynb` | Speckle temporal decorrelation and lifetimes, integrator vs predictive, on the pre-processed stacks |
| `plotting_contrast_results.ipynb` | Loads `overall_best.h5` from stage 2 and plots best contrast curves and post-processing gains |
| `satellite_speckle_psf.ipynb` (+ `.py`) | Satellite-speckle photometry for flux calibration, builds `master_psf.npy` and `light_intensities.json` |
| `light_intensities.json` | Output of the above: PSF and satellite-speckle intensities per run, used by stage 2 for scaling |
| `Fits_visualisation.ipynb` | Exports phase screens, pre-processed cubes, ACF maps and PCA residuals to FITS, makes comparison movies |
| `pre_processing_more_and_analysis.ipynb` | Development of the waffle and coronagraph-mask centring, file checks, spatial correlation, raw contrast |
| `Speckle_linearity.ipynb` | Checks speckle intensity linearity between datasets |
| `AO_explainer_movies.ipynb` | Stand-alone HCIPy closed-loop AO simulation that renders explainer movies (atmosphere, WFS, DM, corrected PSF, coronagraph) to `results/animations/ao_explainer/` |
| `SLM_process.py` | Converts phase-screen `.npy` cubes to zero-padded `.mat` files for the GHOST SLM |

### Simulation (HCIPy)

| File | One line |
|---|---|
| `HCIPY_tests.ipynb` | First HCIPy approach: VLT pupil, PSF, aberrations, companion, vortex/Lyot coronagraphs |
| `HCIPY_post_processing.ipynb` | Simulated frames from int/pred phase screens, contrast curves with applefy (imports top-level `functions`) |

### Legacy 🕰 (written before the `src/` layout, kept for reference)

| File | One line |
|---|---|
| `pre_processing_initial_setting_up.ipynb` | Source stability, display-setting effects, first centring and cropping tests, speckle diversity |
| `GHOST_post_processing.ipynb` | First contrast curves and contrast grids on GHOST data, applefy tutorial cells |
| `TTest_issue.ipynb` | Investigation of the t-test in contrast grids, PCA vs cADI, `num_iter` and `safety_margin` sweeps |
| `post_processing_contrast.py` | Script predecessor of stage 2 using plain applefy `Contrast` |
| `post_processing_fastcontrast.py` | Same with the `ContrastFast` extension |
| `post_processing_fastcontrast_loop.py` | Same, looped over datasets and GPU reductions |
| `utils/functions.py` | Big helper module for the legacy notebooks (HCIPy frames, fake planets, contrast, plotting) |
| `utils/HCIPY.py` | HCIPy-only subset of `utils/functions.py` |

Note: the root README lists a `view_GHOST_screens.ipynb`; it is not in the folder at the moment.

### `notebooks/results/` 📦 (outputs, not tracked)

| Folder | What is in it |
|---|---|
| `animations/ao_explainer/` | Movies and first/last frames from `AO_explainer_movies.ipynb` |
| `animations/pca_anim/`, `animations/curves/` | Older PCA-component GIFs and contrast-curve PNGs per dataset |
| `applefy_movies/` | Fake-planet and residual movies (cADI vs PCA) from `Fits_visualisation.ipynb` |
| `comparison_movies/` | Log-scale OL / int / pred and raw-frame movies from `Fits_visualisation.ipynb` |
| `fits_cubes/` | FITS cubes (raw, median-subtracted, ACF) per run for viewing in DS9 etc. |
| `contrast_curves/`, `contrast_grid/` | Legacy per-dataset contrast outputs from the old scripts |
| `tutorial/` | Outputs of the applefy tutorial cells |

---

## `data/` 📦 (not tracked, provided separately)

| Folder | What is in it |
|---|---|
| `ghost_images/phase_screens_raw/`, `phase_screens_send/` | Phase screens as `.mat` for the SLM (generated and the ones actually sent) |
| `ghost_images/phase_screens_SLM/<date>/` | Raw `.tif` acquisitions from the bench, one folder per session (e.g. `10_13`, `8_24`) |
| `ghost_images/10_15_20ws/` | Main cleaned stacks: `int_*`/`pred_*` per r0 and wind speed, two batches, plus `master_psf.npy`, `raw_contrast.h5`, `outlier_diff.h5` |
| `ghost_images/10_20ws/`, `10_20ws_2/` | Earlier cleaned stacks (old naming `int_0.5ro_10ws.npy`) |
| `ghost_images/long_7ro_20ws/`, `long_7ro_20ws_focus/` | Long sequences for the temporal decorrelation study, split by key (`1key`, `24key`, `32key`) and frame range |
| `ghost_images/bench/` | Closed-loop / open-loop bench stacks (`CL_1-4.npy`, `OL_7ro_20ws_*key_1-4.npy`) |
| `ghost_images/tests_and_HCIPY/` | Simulated datasets and the applefy beta Pic test cube |
| `JALO_stg1/integrator/`, `predictive/` | Residual phase screens from the AO simulation (input to the HCIPy notebooks) |

---

## Quick "where do I..." 

- **Run stage 1 on new bench data:** edit `src/pipeline_pre_ADI/notebooks/config_pre_process.yaml`, run `src/pipeline_pre_ADI/notebooks/run.py`.
- **Run stage 2 (contrast curves):** copy `src/pipeline_ADI/config_files/template.yaml`, run `python -m pipeline_ADI.contrast_pipeline.cli -c my.yaml` (or `src/pipeline_ADI/run.py` for sweeps).
- **Look at final figures:** `notebooks/plotting_contrast_results.ipynb` (contrast) and `notebooks/temporal_decorrelation_clean.ipynb` (lifetimes).
- **Make the AO explainer movies:** `notebooks/AO_explainer_movies.ipynb`.
