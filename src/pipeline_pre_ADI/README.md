python -m venv env_hci
source env_hci/bin/activate
pip install -r requirements.txt
cd src/pipeline_ADI
pip install -e.


Then in VS Code:
Ctrl+Shift+P
Python: Select Interpreter
Choose /home/aosimul/noah/env_hci/bin/python

# Pre-Processing Pipeline (centering, dark subtraction, outlier rejection)

Prepares raw `.tif` image stacks for the ADI / contrast pipeline: aligns frames, finds the star centre, subtracts a dark, crops, normalises, and rejects bad frames using PCA with a MAD criterion.


Note: I rely on satellite speckle detection in my pre processing routine. I thus had to hard code code coordinates to locate them or at least define a searching area. For reference, I took my data acquisition defining a windowing in the Thorlabs software for the science camera on GHOST, of widths 200 by 200, with un upper left corder at (836, 342). This means the reference satellite speckle I used was located around (102,172) and I also defined a rectangular aperture to block out the center of the image and only show satellite speckles, with assumes that the data acquisition is somewhat centred on the source.


Entry points:

| Function | Purpose |
|---|---|
| `run_pre_processing(img_file, ...)` | Load, mask, align, centre, dark-subtract, crop, normalise. Returns `(stack, normalisation_cnst, speckle_intensity)` |
| `run_outlier_detection(stack_name, stack, ...)` | PCA-based bad-frame rejection and re-normalisation. Returns `(stack, outlier_matrix, normalisation_cnst)` |

Typical use: `run_pre_processing` then `run_outlier_detection` on its output.

---

## Block diagram

```mermaid
flowchart TD
    A["img_file prefix<br/>glob '*.tif'"] --> B["tiff.imread"]
    B --> C["Keep first img_key frames<br/>(optional)"]
    C --> D["masking_function<br/>(optional)"]
    D --> CS

    subgraph CS["center_stack"]
        E["rect_mask(stack)"] --> F["align_stack<br/>phase_cross_correlation (upsampled)<br/>+ Fourier shift"]
        F --> G["estimate_center"]
        G --> H["Sub-pixel shift to nearest integer pixel<br/>(spline order 3)"]
    end

    subgraph EC["estimate_center methods"]
        G1["waffle<br/>find_satellite_speckles"]
        G2["masked_annulus<br/>detect_coron_mask_and_fit_circle"]
        G3["center_of_mass<br/>(search window around ref PSF peak)"]
        G4["gaussian<br/>fit_2d_gaussian"]
        G5["cross_correlation<br/>estimate_center_correlation"]
        G6["fixed<br/>user coords"]
    end

    G -.-> G1
    G -.-> G2
    G -.-> G3
    G -.-> G4
    G -.-> G5
    G -.-> G6

    H --> I{"dark_file?"}
    I -- yes --> J["Load darks, centre at (yc, xc)<br/>master dark = median<br/>subtract from stack"]
    I -- no --> K
    J --> K["Speckle intensity<br/>max of rect_aperture_mask(median, 102, 172, size=20)"]
    K --> L["crop_stack<br/>radius or largest odd square"]
    L --> M["Clip at eps, cast to float32"]
    M --> N{"normalisation?"}
    N -- yes --> O["Divide by global max"]
    N -- no --> P["constant = 1"]
    O --> Q["Output: stack, normalisation_cnst, speckle_intensity"]
    P --> Q

    Q --> R["run_outlier_detection"]

    subgraph OD["run_outlier_detection"]
        R1{"mask?"} -- yes --> R2["stack * mask"]
        R1 -- no --> R3["use stack as is"]
        R2 --> S["pca_frame_selection"]
        R3 --> S
        S --> S1["Flatten frames, mean-centre pixels"]
        S1 --> S2["PCA (randomized SVD, n_components)"]
        S2 --> S3["Normalise each component by its max abs"]
        S3 --> S4["median + MAD per component"]
        S4 --> S5["Frame is bad if any component<br/>deviates > mad_threshold * MAD"]
        S5 --> T["Keep good frames of the unmasked stack"]
        T --> U{"normalisation?"}
        U -- yes --> V["Re-normalise by new global max"]
        U -- no --> W["constant = 1"]
        V --> X["Clean stack + normalisation constant"]
        W --> X
        S5 --> Y{"PCA_analysis?"}
        Y -- yes --> Z["plot_pca_frame_selection<br/>save PNG"]
    end

    R --> R1
    X --> ZA["Clean stack, outlier_matrix, normalisation_cnst"]

```

---

## Stages in detail

### 1. `run_pre_processing`
1. Read all `{img_file}*.tif` files; optionally keep the first `img_key` frames.
2. Apply `masking_function`, if given, to the whole sequence.
3. **Centre and align** (`center_stack`): frames are aligned to a reference built from `rect_mask(stack)` using `phase_cross_correlation` and a Fourier shift (preserves speckle statistics better than spline interpolation). The `rect_mask(stack)` was defined to only keep the satellite speckles which are reliable features on the GHOST bench. The star centre is then estimated and the stack is shifted so the centre lands on an integer pixel.
4. **Dark subtraction** (if `dark_file`): darks are shifted to the same centre (using `method="fixed"`), median-combined into a master dark and subtracted.
5. **Speckle intensity**: maximum of a small rectangular aperture on the median image. The aperture was defined according to the coordinates and the windowing I performed my data acquisition with.
6. **Crop** around the centre (using a defined size `radius`, or the largest square that fits).
7. **Normalise**: pixels clipped to at least `eps` to avoid zero values in the image, converted to float32, divided by the global maximum if `normalisation=True`.

### 2. Centre estimation methods (`method=`)

| Method | Idea | Notes |
|---|---|---|
| `waffle` (default in `center_stack`/`run_pre_processing`) | Fit centre from satellite (waffle) speckles in the mean image | `n_satellites` |
| `masked_annulus` (default in `estimate_center`) | Fit a circle to the coronagraph mask edge in the median image | |
| `center_of_mass` | Centroid of the median image, optionally in a window around a reference-PSF peak | `search_radius`, `smooth_sigma` |
| `gaussian` | 2D Gaussian fit on the median image | `search_radius` clamps the result |
| `cross_correlation` | Iterative rotational cross-correlation | needs `estimate_center_correlation`, uses `upsample_factor`, `max_iter`, `tol` |
| `fixed` | Use given `coords=(yc, xc)` | |

`reference_stack` may be an array or a path prefix (`{prefix}*.tif`, median-combined, Gaussian-smoothed).

#### Centering block diagram (`estimate_center` and `centering.py`)

```mermaid
flowchart TD
    IN["aligned stack (N, ny, nx)<br/>+ method, coords, reference_stack,<br/>search_radius, smooth_sigma"] --> V["Validate: stack must be 3D"]
    V --> REF["_prepare_ref<br/>path prefix: glob *.tif, or array<br/>median over frames, Gaussian smooth<br/>= ref_psf (or None)"]
    REF --> SW{"method"}

    SW -- "fixed" --> FX["return coords (yc, xc)<br/>error if coords is None"]

    SW -- "waffle" --> W0["mean of stack"]
    subgraph WF["waffle: find_satellite_speckles"]
        W1["rect_mask<br/>RectangularAnnulus around centre_subpixel<br/>(outer 170 px, inner 135 px)"] --> W2["find_peaks<br/>box_size = 2*min_distance + 1"]
        W2 --> W3["sort peaks by brightness"]
        W3 --> W4["greedy selection:<br/>skip peaks closer than min_distance (30 px)<br/>until n_satellites found"]
        W4 --> W5["centre = mean (x, y)<br/>of selected speckles"]
    end
    W0 --> W1
    W5 --> OUT

    SW -- "masked_annulus" --> M0["median of stack"]
    subgraph MA["masked_annulus: detect_coron_mask_and_fit_circle"]
        M1["Gaussian smooth<br/>(sigma = 10)"] --> M2["threshold at 0.7 * max"]
        M2 --> M3["label connected regions<br/>keep the largest"]
        M3 --> M4["boundary = mask minus eroded mask<br/>(2 iterations)"]
        M4 --> M5{"at least 10<br/>boundary points?"}
        M5 -- no --> M6["use all mask pixels"]
        M5 -- yes --> M7["algebraic least-squares circle fit"]
        M6 --> M7
        M7 --> M8["centre (yc, xc), radius"]
    end
    M0 --> M1
    M8 --> OUT

    SW -- "center_of_mass" --> C0["weights = median of stack"]
    C0 --> C1{"ref_psf available?"}
    C1 -- yes --> C2["peak of ref_psf (yc0, xc0)<br/>window = peak +/- search_radius"]
    C2 --> C3["subtract window min<br/>centroid of window"]
    C3 --> C4["clip result to<br/>peak +/- search_radius"]
    C1 -- no --> C5["subtract global min<br/>centroid of whole image"]
    C4 --> OUT
    C5 --> OUT

    SW -- "gaussian" --> G0["median of stack"]
    subgraph GA["gaussian: fit_2d_gaussian"]
        G1["optional bbox around ref_psf peak<br/>(if search_radius set)"] --> G2["initial guess:<br/>amplitude, centroid, sigma, offset"]
        G2 --> G3["curve_fit rotated 2D Gaussian"]
        G3 -- "fit fails" --> G4["fallback: centre of mass"]
        G3 -- "ok" --> G5["centre (yc, xc) in full-image coordinates"]
        G4 --> G5
    end
    G0 --> G1
    G5 --> G6["clip to ref_psf peak +/- search_radius<br/>(if ref_psf and search_radius)"]
    G6 --> OUT

    SW -- "cross_correlation" --> X0["estimate_center_correlation"]
    subgraph XC["cross_correlation: iterative rotational method"]
        X1["ref_frame = median of first nframes<br/>(min(1000, N))"] --> X2["start: ref_psf peak,<br/>else geometric centre"]
        X2 --> X3["square sub-image around current centre<br/>(largest symmetric half-width)"]
        X3 --> X4["rotate sub-image by 90 degrees"]
        X4 --> X5["phase_cross_correlation(sub, rotated)<br/>sub-pixel shift"]
        X5 --> X6["convert shift to (dy, dx) correction<br/>update centre"]
        X6 --> X7["clip to ref_psf peak +/- search_radius"]
        X7 --> X8{"correction < tol<br/>or max_iter reached?"}
        X8 -- no --> X3
        X8 -- yes --> X9["centre (yc, xc)"]
    end
    X0 --> X1
    X9 --> OUT

    OUT["(yc, xc) as floats"] --> CS["back in center_stack:<br/>round to nearest integer pixel,<br/>shift stack by the sub-pixel remainder (order 3)"]
```


### 3. `run_outlier_detection`
Frames are flattened and decomposed with PCA. In each component the median and MAD are computed. A frame is rejected if it lies beyond `mad_threshold` MADs in **any** of the `n_components` components. Rejection is decided on the (optionally masked) stack, but the output keeps the unmasked good frames, re-normalised. With `PCA_analysis=True`, a PC1 vs PC2 scatter plot is saved to `{PCA_plot_output}/{stack_name}.png`.

### 4. Helpers
- `zoom_to_peak(img, radius)`: crop a square around the brightest pixel, returns `(cropped, peak_coords)`.
- `radial_profile_pre(image, binsize)`: azimuthal mean and standard deviation about the geometric image centre.

## Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `method` | `"waffle"` | Centre estimation method |
| `radius` | `None` | Half-width of the crop (pixels) |
| `search_radius` | 7 | Allowed distance from the reference PSF peak |
| `smooth_sigma` | 3 | Gaussian smoothing of the reference PSF |
| `upsample_factor` | 100 | Sub-pixel precision for cross-correlation centering |
| `n_satellites` | 2 | Waffle spots used |
| `eps` | 1e-9 | Floor value, keeps pixels strictly positive |
| `n_components` | 5 | PCA components for outlier detection |
| `mad_threshold` | 5.0 | Rejection threshold in MADs |

## Requirements

numpy, scipy, scikit-image, scikit-learn, photutils, tifffile, matplotlib, pandas, `applefy`, and the project modules `utils.statistics`, `utils.naming` (`titling`), `pipeline_pre_ADI.pre_process.centering` (`rect_mask`, `rect_aperture_mask`, `find_satellite_speckles`, `detect_coron_mask_and_fit_circle`, `fit_2d_gaussian`, `estimate_center_correlation`).

## Known issues / things to check

- The speckle-intensity aperture is hard-coded: `rect_aperture_mask(median, 102, 172, size=20)`. The aperture was defined according to the coordinates and the windowing I performed my data acquisition with. It corresponds to the lower center satellite speckle, with a windowing of 200 by 200, with un upper left corder at (836, 342) on the GHOST science camera.
- Normalisation happens twice: once at the end of `run_pre_processing` and again after outlier rejection. The constant returned by `run_pre_processing` therefore does not convert back to raw counts on its own. Multiply both constants if you need absolute flux.
- `center_stack` docstring does not list `masked_annulus` or `gaussian`.
- `shift(..., mode="constant", cval=0)` after centring introduces zero-valued edges. Cropping removes them only if `radius` is small enough.
- The `rect_mask(stack)` mask used for alignement was defined to only keep the satellite speckles which are reliable features on the GHOST bench. It also corresponds to my data acquisition, with a windowing of 200 by 200, with un upper left corder at (836, 342) on the GHOST science camera.

Notes on the centering code:

- `find_satellite_speckles` defaults to `n_satellites=6`, but `estimate_center` passes `n_satellites=2` (the `run_pre_processing` default), so the centre is the midpoint of the two brightest well-separated peaks. This assumes they are a diametrically opposite waffle pair.
- `find_peaks(..., mask=~np.isfinite(masked_image))` masks nothing, because the image is finite everywhere. The annulus restriction comes from `rect_mask` zeroing the rest.
- If `find_peaks` finds nothing, `find_satellite_speckles` returns `[]` (one value) where the caller unpacks three, so it would fail with an unpacking error. If peaks exist but fewer than expected are found, the centre is still the mean of those found.
- `detect_coron_mask_and_fit_circle` is documented as detecting a *dark* mask, but the code keeps pixels *above* `0.7 * max` of the smoothed image. This works if the image is inverted or the bright ring is what is wanted. Check this matches your data. The `max_area` sanity check is commented out.
- `centering.py` defines its own `centroid_com` returning `(y, x)`. Because it is star-imported after `photutils.centroids.centroid_com`, it silently replaces the photutils version, which returns `(x, y)`. The `center_of_mass` branch relies on `(y, x)`, so keep the import order.
- `cross_correlation` is marked "conceptual, not very robust" in its own docstring.


