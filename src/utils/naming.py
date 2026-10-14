"""Run naming, name parsing/titling and data loading, in one module.

Layout (each block only depends on the ones above it):
    1. Naming    - build and iterate run tags
    2. Parsing   - split/reorder arbitrary name strings, make plot titles
    3. Loading   - read stacks, raw .tif files and contrast tables
"""
import glob
import re
from pathlib import Path

import numpy as np
import tifffile as tiff

# ----------------------------------------------------------------------------
# 1. Naming
# ----------------------------------------------------------------------------


def run_name(ro=None, ws=None, key=None, count=None):
    """Run tag used in file names, e.g. '7ro_20ws_1key_1-4'. Missing parts are skipped."""
    parts = []
    if ro is not None:
        parts.append(f"{ro}ro")
    if ws is not None:
        parts.append(f"{ws}ws")
    if key is not None:
        parts.append(f"{key}key")
    if count is not None:
        parts.append(str(count))
    return "_".join(parts)


def iter_runs(ros, wss, keys=(None,), counts=(None,)):
    """Yield one dict(ro, ws, key, count) per combination."""
    for count in counts:
        for key in keys:
            for ro in ros:
                for ws in wss:
                    yield dict(ro=ro, ws=ws, key=key, count=count)


# ----------------------------------------------------------------------------
# 2. Parsing / reordering / titling
# ----------------------------------------------------------------------------

_PRIORITIES = {"int": "Integrative AO:", "pred": "Predictive AO:",
               "OL": "Open Loop:", "CL": "Closed Loop:"}
_RO_RE = re.compile(r"(\d*\.?\d+)(?:r0|ro)")
_WS_RE = re.compile(r"(\d+)(?:windspeed|ws)")
_KEY_RE = re.compile(r"(?:(\d+)key|key(\d+))")


def section_name(s):
    """Split a name into (priority, arcsec, windspeed, key, rest).

    priority: 'int' | 'pred' | 'OL' | 'CL'
    arcsec:   '<n>ro' (decimals are converted to tenths: '0.7r0' -> '7ro')
    windspeed:'<n>ws' (from '<n>windspeed' or '<n>ws')
    key:      '<n>key' (from '<n>key' or 'key<n>')
    rest:     list of all other parts, in order of appearance
    """
    priority = arcsec = windspeed = key = None
    rest = []

    for part in s.split("_"):
        if part in _PRIORITIES:
            priority = part
        elif m := _RO_RE.fullmatch(part):
            num = m.group(1)
            num = int(float(num) * 10) if "." in num else int(num)
            arcsec = f"{num}ro"
        elif m := _WS_RE.fullmatch(part):
            windspeed = f"{int(m.group(1))}ws"
        elif m := _KEY_RE.fullmatch(part):
            key = f"{int(m.group(1) or m.group(2))}key"
        else:
            rest.append(part)

    return priority, arcsec, windspeed, key, rest


def reorder_name(s):
    """Return (full, special, rest) with the order priority, ro, ws, key, rest."""
    *special, rest = section_name(s)
    special = [p for p in special if p is not None]
    return (
        "_".join(special + rest),
        "_".join(special),
        "_".join(rest),
    )


def titling(s, show_key=False):
    """Human-readable plot title, e.g. 'Predictive AO: 0.7 arcsec, 20 m/s'."""
    priority, arcsec, windspeed, key, rest = section_name(s)

    bits = []
    if priority:
        bits.append(_PRIORITIES[priority])
    if arcsec:
        bits.append(f"{int(arcsec[:-2]) / 10:g} arcsec,")
    if windspeed:
        bits.append(f"{int(windspeed[:-2])} m/s")
    if show_key and key:
        bits.append(f", Key: {key[:-3]}")

    title = " ".join(bits).replace(" ,", ",")
    if rest:
        title += f"\n({' '.join(rest)})"
    return title


# ----------------------------------------------------------------------------
# 3. Loading
# ----------------------------------------------------------------------------


def load_stack(folder, algo, dtype=None, **run):
    """Load '<folder>/<algo>_<run_name>.npy'. `run` takes ro, ws, key, count."""
    name = "_".join(p for p in (algo, run_name(**run)) if p)
    stack = np.load(Path(folder) / f"{name}.npy")
    return stack if dtype is None else stack.astype(dtype)


def list_raw_files(folder):
    """Split the .tif files of `folder` into (science, psf) lists."""
    files = sorted(glob.glob(str(Path(folder) / "*.tif")))
    is_psf = lambda f: "no_coro" in Path(f).name
    science = [f for f in files if "coro" in Path(f).name and not is_psf(f)]
    return science, [f for f in files if is_psf(f)]


def read_tif_stack(path, n_frames=None):
    """Read the first `n_frames` frames of a .tif file (all frames if None)."""
    return tiff.imread(path, key=slice(0, n_frames))


def read_raw_contrast(store, tag, level="pixel_rad"):
    """(radius, raw_contrast) of entry '/<tag>' in an open pd.HDFStore.

    `level` is 'pixel_rad' (pixels) or 'pixel_rad_fwhm' (units of FWHM)."""
    df = store[f"/{tag}"]
    return df.index.get_level_values(level), df["raw_contrast"]