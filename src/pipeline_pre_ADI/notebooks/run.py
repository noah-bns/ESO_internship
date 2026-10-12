"""Ghost-image preprocessing pipeline.

Steps per science image:
  1. align / centre / crop        (run_pre_processing)
  2. outlier detection via PCA    (run_outlier_detection)
  3. save cleaned stack (.npy)
  4. process the matching PSF     (every N-th image)
  5. optional raw-contrast profile

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --set modes.save_outputs=false
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from photutils.aperture import CircularAperture

# Provides run_pre_processing, run_outlier_detection, radial_profile_pre,
# calculate_fwhm (kept as star imports exactly as in the original script).
from pipeline_pre_ADI.pre_process.pre_processing import *  # noqa: F401,F403
from pipeline_pre_ADI.pre_process.plot_pre_processing import *  # noqa: F401,F403
from utils.naming import reorder_name

log = logging.getLogger("pipeline")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def input_dir(self) -> Path:
        return Path(self.raw["paths"]["input_dir"])

    @property
    def data_dir(self) -> Path:
        return Path(self.raw["paths"]["data_root"]) / self.raw["run_name"]

    @property
    def analysis_dir(self) -> Path:
        return Path(self.raw["paths"]["analysis_root"]) / self.raw["run_name"]

    @property
    def speckle_dir(self) -> Path:
        return self.analysis_dir / self.raw["outputs"]["speckle_diversity_subdir"]


def load_config(path: str, overrides: list[str]) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[leaf] = yaml.safe_load(value)  # parses true/false/numbers/null
    return Config(raw)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def prepare_directories(cfg: Config) -> None:
    if cfg["modes"]["start_over"] and cfg.data_dir.exists():
        shutil.rmtree(cfg.data_dir)
        log.warning("Removed existing directory: %s", cfg.data_dir)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.speckle_dir.mkdir(parents=True, exist_ok=True)


def collect_files(cfg: Config) -> tuple[list[str], list[str]]:
    sel = cfg["file_selection"]
    files = sorted(glob.glob(str(cfg.input_dir / sel["glob"])))
    name = lambda f: Path(f).name
    img_files = [
        f for f in files
        if sel["science_contains"] in name(f) and sel["science_excludes"] not in name(f)
    ]
    psf_files = [f for f in files if sel["psf_contains"] in name(f)]
    if sel["limit"] is not None:
        img_files = img_files[: sel["limit"]]
    log.info("Science images: %d | PSF images: %d", len(img_files), len(psf_files))
    return img_files, psf_files


def unique_name(path: str, root_counts: Counter, running: Counter, prefix: str = "") -> str:
    """Return the output stem; add a running index when a root occurs several times."""
    _, root, _ = reorder_name(Path(path).name)
    if root_counts[root] > 1:
        running[root] += 1
        return f"{prefix}{root}_{running[root]}"
    return f"{prefix}{root}"


def input_path(cfg: Config, name: str) -> str:
    return str(cfg.input_dir / name)


def is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def compute_uncorrelated_psf(cfg: Config):
    p, pre = cfg["preprocessing"]["uncorrelated_psf"], cfg["preprocessing"]
    psf, norm_const, _ = run_pre_processing(
        img_file=input_path(cfg, p["img_file"]),
        img_key=p["img_key"],
        dark_file=input_path(cfg, p["dark_file"]),
        dark_key=p["dark_key"],
        radius=cfg["geometry"]["radius"],
        normalisation=p["normalisation"],
        method=pre["method"],
        n_satellites=pre["n_satellites"],
    )
    if cfg["modes"]["save_outputs"]:
        np.save(cfg.data_dir / "psf_uncorrelated.npy", np.mean(psf, axis=0))
    return norm_const


def process_science(cfg: Config, img_file: str, stem: str, outlier_mask):
    pre, sci, od = cfg["preprocessing"], cfg["preprocessing"]["science"], cfg["outlier_detection"]
    processed, norm_1, speckle_intensity = run_pre_processing(
        img_file=input_path(cfg, Path(img_file).stem),
        img_key=sci["img_key"],
        dark_file=input_path(cfg, sci["dark_file"]),
        dark_key=sci["dark_key"],
        radius=cfg["geometry"]["radius"],
        method=pre["method"],
        n_satellites=pre["n_satellites"],
        normalisation=sci["normalisation"],
    )
    cleaned, outlier_matrix, norm_2 = run_outlier_detection(
        stack_name=stem,
        sci_img_processed=processed,
        n_components=od["n_components"],
        mad_threshold=od["mad_threshold"],
        PCA_analysis=od["pca_analysis"],
        PCA_plot_output=str(cfg.speckle_dir),
        normalisation=od["normalisation"],
        mask=outlier_mask,
    )
    mean_clean = np.mean(cleaned, axis=0)
    diff = np.mean(processed, axis=0) - mean_clean

    if cfg["modes"]["save_outputs"]:
        np.save(cfg.data_dir / f"{stem}.npy", cleaned)
    return mean_clean, diff, norm_1, norm_2, speckle_intensity


def process_psf(cfg: Config, psf_file: str, out_stem: str):
    pre, p = cfg["preprocessing"], cfg["preprocessing"]["psf"]
    processed, _, _ = run_pre_processing(
        img_file=input_path(cfg, Path(psf_file).stem),
        img_key=p["img_key"],
        dark_file=input_path(cfg, p["dark_file"]),
        dark_key=p["dark_key"],
        radius=cfg["geometry"]["radius"],
        method=pre["method"],
        n_satellites=pre["n_satellites"],
        normalisation=p["normalisation"],
    )
    psf_med = np.mean(processed, axis=0)
    if cfg["modes"]["save_outputs"]:
        np.save(cfg.data_dir / f"{out_stem}.npy", psf_med)
    return calculate_fwhm(psf_med)


def save_raw_contrast(cfg, stem, mean_clean, fwhm, norm_1, norm_2, norm_uncorr):
    t = cfg["timing"]
    pixel_rad, y_noncoro, _ = radial_profile_pre(mean_clean)
    amplitude = norm_1 / norm_2 / norm_uncorr * t["dit_psf"] / t["dit_img"]
    rad_fwhm = pixel_rad / fwhm if fwhm is not None else np.full_like(pixel_rad, np.nan, dtype=float)
    df = pd.DataFrame(
        {"raw_contrast": amplitude * y_noncoro},
        index=pd.MultiIndex.from_arrays([pixel_rad, rad_fwhm], names=["pixel_rad", "pixel_rad_fwhm"]),
    )
    df.to_hdf(cfg.analysis_dir / cfg["outputs"]["raw_contrast"], key=stem, mode="a")


def stack_repeats(cfg: Config) -> None:
    """Concatenate <name>_<a>..<name>_<b> into <name>_<a>-<b>.npy, then delete the parts."""
    st = cfg["stacking"]
    if not st["enabled"]:
        return
    save = cfg["modes"]["save_outputs"]

    groups: dict[str, dict[int, Path]] = {}
    for f in cfg.data_dir.glob("*.npy"):
        m = re.fullmatch(r"(.+)_(\d+)", f.stem)
        if m and not m.group(1).startswith(st["skip_prefix"]):
            groups.setdefault(m.group(1), {})[int(m.group(2))] = f

    for prefix, numbered in sorted(groups.items()):
        for lo, hi in st["groups"]:
            numbers = range(lo, hi + 1)
            label = f"{prefix}_{lo}-{hi}"
            if not all(n in numbered for n in numbers):
                log.info("Skipping %s: missing files", label)
                continue

            parts = [numbered[n] for n in numbers]
            log.info("Stacking %s: %s", label, [p.name for p in parts])
            if not save:
                continue

            arrays = [np.load(p) for p in parts]
            stacked = np.concatenate(arrays, axis=0)
            out_file = cfg.data_dir / f"{label}.npy"
            np.save(out_file, stacked)

            # verify before deleting anything
            expected = sum(a.shape[0] for a in arrays)
            if np.load(out_file, mmap_mode="r").shape[0] != expected:
                raise RuntimeError(f"Verification failed for {out_file}; originals kept")
            log.info("Saved %s: shape %s", out_file.name, stacked.shape)

            if st["delete_originals"]:
                for p in parts:
                    p.unlink()
                log.info("Deleted %d originals for %s", len(parts), label)


def write_summaries(cfg: Config, light_intensities: dict, diffs: dict) -> None:
    out = cfg["outputs"]
    with open(cfg.analysis_dir / out["light_intensities"], "w") as f:
        json.dump(light_intensities, f, indent=4)
    with h5py.File(cfg.analysis_dir / out["outlier_diff"], "a") as f:
        for key, value in diffs.items():
            if key in f:
                del f[key]
            f.create_dataset(key, data=value)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(cfg: Config) -> None:
    prepare_directories(cfg)
    img_files, psf_files = collect_files(cfg)

    root_name = lambda f: reorder_name(Path(f).name)[1]
    img_root_counts = Counter(root_name(f) for f in img_files)
    psf_root_counts = Counter(root_name(f) for f in psf_files)
    img_counter, psf_counter = Counter(), Counter()

    g = cfg["geometry"]
    outlier_mask = CircularAperture((g["radius"], g["radius"]), g["mask_radius"]).to_mask(method="center")

    raw_contrast = cfg["modes"]["compute_raw_contrast"]
    norm_uncorr = compute_uncorrelated_psf(cfg) if raw_contrast else None

    every_n = cfg["file_selection"]["psf_every_n_images"]
    psf_count, fwhm = 0, None
    diffs, light_intensities = {}, {}

    for i, img_file in enumerate(img_files):
        stem = unique_name(img_file, img_root_counts, img_counter)
        log.info("[%d/%d] %s -> %s", i + 1, len(img_files), Path(img_file).name, stem)

        mean_clean, diff, norm_1, norm_2, speckle_intensity = process_science(
            cfg, img_file, stem, outlier_mask
        )
        diffs[stem] = diff

        # Matching PSF (every N-th science image)
        if psf_files and i % every_n == 0 and psf_count < len(psf_files):
            psf_file = psf_files[psf_count]
            psf_stem = unique_name(psf_file, psf_root_counts, psf_counter, prefix="psf_")
            fwhm = process_psf(cfg, psf_file, psf_stem)
            psf_count += 1

        if raw_contrast:
            save_raw_contrast(cfg, stem, mean_clean, fwhm, norm_1, norm_2, norm_uncorr)

        light_intensities[stem] = {
            "satellite": None if is_nan(speckle_intensity) else float(speckle_intensity)
        }
        plt.close("all")

    write_summaries(cfg, light_intensities, diffs)
    stack_repeats(cfg)
    log.info("Done. Data: %s | Analysis: %s", cfg.data_dir, cfg.analysis_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. modes.save_outputs=false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.set)
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    # keep a copy of the exact config next to the results for reproducibility
    with open(cfg.analysis_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False)
    run(cfg)


if __name__ == "__main__":
    main()