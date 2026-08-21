
import numpy as np
from pathlib import Path
import warnings
import os, glob
from multiprocessing import cpu_count
import importlib
import time
import gc
import pandas as pd


# images
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams["hatch.linewidth"] = 0.5  # previous pdf hatch linewidth
from applefy.utils.positions import center_subpixel
from matplotlib.patches import Circle


# scientific libraries
import pipeline_pre_ADI.library.pre_processing as pre
importlib.reload(pre)
from pipeline_pre_ADI.library.pre_processing import *
import pipeline_pre_ADI.library.statistics as stat
importlib.reload(stat)
from pipeline_pre_ADI.library.statistics import *





def plot_outlier_images_by_pc(
    cube,
    outlier_matrix,
    cmap="inferno",
):
    """
    Plot all outlier images grouped by PCA component.

    Parameters
    ----------
    cube : ndarray
        Image cube (n_frames, ny, nx)
    outlier_matrix : ndarray
        Boolean array (n_frames, n_components)
    """

    n_components = outlier_matrix.shape[1]

    # Global normalization over ALL outlier images
    all_indices = np.where(np.any(outlier_matrix, axis=1))[0]

    if len(all_indices) == 0:
        print("No outliers found.")
        return

    data = cube[all_indices]

    # Log scale requires positive values
    vmin = 4e-3 #np.percentile(data[data > 0], 1)
    vmax = np.percentile(data[data > 0], 99.5)

    norm = LogNorm(vmin=vmin, vmax=vmax)


    for pc in range(n_components):

        indices = np.where(outlier_matrix[:, pc])[0]

        if len(indices) == 0:
            continue

        ncols = 6
        nrows = int(np.ceil(len(indices) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3*ncols, 3*nrows),
        )

        axes = np.atleast_1d(axes).ravel()

        im = None

        for ax, idx in zip(axes, indices):

            # avoid zero values for LogNorm
            image = np.maximum(cube[idx], vmin)

            im = ax.imshow(
                image,
                origin="lower",
                cmap=cmap,
                norm=norm,
            )

            ax.set_title(f"Frame {idx}")
            ax.axis("off")


        # remove unused panels
        for ax in axes[len(indices):]:
            ax.remove()


        # one colorbar for this PC group
        cbar = fig.colorbar(
            im,
            ax=axes[:len(indices)],
            fraction=0.02,
            pad=0.02,
        )

        cbar.set_label("Intensity (log scale)")


        fig.suptitle(
            f"Outliers detected by PC{pc+1} "
            f"({len(indices)} frames)",
            ha = 'center',
            fontsize=16
        )
        #plt.tight_layout()
        plt.show()



def raw_contrast(unaberrated_PSF,
                      mean_psfs,
                      cubes_coro,
                      DIT_psf, 
                      DIT_sc_frame,
                      labels = ['Classic AO', 'Predicitve AO' ],
                      binsize = 1,
                      plot = False,
                      ):

    #outputs
    seps = []
    raw_contrasts = []
    strehls = []

    # normalise the exposure times
    factor = DIT_psf / DIT_sc_frame

    # Normalize everything by the same peak
    peak = unaberrated_PSF.max()
    r0, y_unaber, _ = radial_profile_pre(unaberrated_PSF / peak, binsize)

    colors = [
        '#7F3C8D', '#11A579', '#3969AC', '#F2B701',
        '#E73F74', '#80BA5A', '#E68310', '#008695',
        '#CF1C90', '#F97B72', '#4B4B8F', '#A5AA99'
    ]

    for (i, (mean_psf, cube_coro)) in enumerate(zip(mean_psfs, cubes_coro)):        
        mean_psf_coro = np.mean(cube_coro, axis = 0) * factor

        _, y_coro, _ = radial_profile_pre(mean_psf_coro / peak, binsize)

        strehl = mean_psf.max() / peak
        r = np.float64(r0) / calculate_fwhm(mean_psf)

        seps.append(r)
        raw_contrasts.append(y_coro)
        strehls.append(strehl)

        if plot == True:
            label = labels[i]

            _, y_noncoro, _ = radial_profile_pre(mean_psf / peak, binsize)
        
            if i ==0:
                plt.plot(r,
                        y_unaber,
                        color='k',
                        label='Unaberrated')
                plt.plot(r, y_noncoro, '--',  label='Non-coronagraphic', color = 'black', lw = 2)
                plt.plot(r, y_coro,':', label='Coronagraphic', color = 'black', lw = 2)      
            plt.plot(r, y_noncoro, '--', color = colors[i], lw = 2)
            plt.plot(r, y_coro, ':', color = colors[i], lw = 2)
            plt.scatter(0, 0, 
                        marker = 'o', 
                        color = colors[i], 
                        label = (label + f' Strehl = {float(strehl):.2f}')
                        )
            plt.yscale('log')
            plt.xlim(left =0)
            plt.ylim(1e-5, 1)

            plt.xlabel("Angular separation [FWHM]")
            plt.ylabel("Raw contrast")

            plt.legend()
            plt.tight_layout()
    
    return seps, raw_contrasts, strehls

def plot_raw_contrast(unaberrated_PSF,
                      mean_psfs_coro,
                      mean_psfs,
                      binsize = 1):

    #rad_to_arcsec = 206265

    # Normalize everything by the same peak
    peak = unaberrated_PSF.max()
    r0, y_unaber, _ = radial_profile_pre(unaberrated_PSF / peak, binsize)


    #plt.figure(figsize=(6,4))
    colors = ['#F97B72', '#4B4B8F', 'blue']

    # plt.plot(r0,
    #         y_unaber,
    #         color='k',
    #         label='Unaberrated')
    
    for (i, (mean_psf_stat, mean_psf_stat_coro)) in enumerate(zip(mean_psfs, mean_psfs_coro)):
        _, y_noncoro, _ = radial_profile_pre(mean_psf_stat / peak, binsize)
        _, y_coro, yerr = radial_profile_pre(mean_psf_stat_coro / peak, binsize)

        strehl = mean_psf_stat.max() / peak
        r = np.float64(r0) / calculate_fwhm(mean_psf_stat)

        label = 'Classic AO' if i==0 else 'Predicitve AO' 

        if i ==0:
            plt.plot(r,
                    y_unaber,
                    color='k',
                    label='Unaberrated')
            plt.plot(r, y_noncoro, '--',  label='Non-coronagraphic', color = 'black', lw = 2)
            plt.plot(r, y_coro,':', label='Coronagraphic', color = 'black', lw = 2)      
        plt.plot(r, y_noncoro, '--', color = colors[i], lw = 2)
        plt.plot(r, y_coro, ':', color = colors[i], lw = 2)
        plt.scatter(0, 0, marker = 'o', color = colors[i], label = (label + f' Strehl = {float(strehl):.2f}'))

    plt.yscale('log')
    plt.xlim(left =0)
    plt.ylim(3e-6, 1)

    plt.xlabel("Angular separation [FWHM]")
    plt.ylabel("Raw contrast")

    plt.legend()
    plt.tight_layout()

    

def plot_symmetric_intensities_single( 
        data_cube, ax_img, ax_int, title, angle=45.0, coro_rad=8, st2_rad=32, dr=5.0, crop=20, aperture_radius=1.5, r_aperture = 0.5):
    """
    Plot the image and symmetric-intensity relation for one dataset.

    Parameters
    ----------
    data_cube : ndarray
        Intensity cube with shape (n_samples, ny, nx).

    ax_img : matplotlib.axes.Axes
        Axes object for the image plot.

    ax_int : matplotlib.axes.Axes
        Axes object for the intensity plot.

    title : str
        Title for the figure.

    angle : float
        Sampling angle in degrees.

    coro_rad : float
        Coronagraph radius in pixels.

    st2_rad : float
        Outer sampling radius in pixels.

    dr : float
        Radial sampling step in pixels.

    crop : int
        Number of pixels cropped from each image edge.

    aperture_radius : float
        Radius of the aperture markers in pixels.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """

    # ========================================================
    # Extract symmetric intensities
    # ========================================================
    (
        (x1, y1),
        left_int,
        (x2, y2),
        symmetric_int,
    ) = symmetric_intensities(
        data_cube[:, crop:-crop, crop:-crop],
        angle,
        coro_rad,
        st2_rad,
        dr=dr,
        r_aperture = r_aperture
    )

    # ========================================================
    # Radial positions
    # ========================================================

    radii = np.arange(
        coro_rad + 0.5 * dr + r_aperture,
        st2_rad - r_aperture,
        (dr + 2 * r_aperture)
    )

    # ========================================================
    # COLORMAP
    # ========================================================
    # Colors are now based purely on the sample index,
    # NOT on radial distance.
    cmap = plt.get_cmap("coolwarm")

    size = len(radii) * len(angle) if isinstance(angle, list) else len(radii)
    colors = [
        cmap(i / max(size - 1, 1))
        for i in range(size)
    ]

    # ========================================================
    # IMAGE
    # ========================================================
    image = data_cube[
        0,
        crop:-crop,
        crop:-crop,
    ]

    cy, cx = center_subpixel(image)

    # --------------------------------------------------------
    # Image intensity normalization
    # --------------------------------------------------------
    # Ignore zero / negative values because LogNorm requires
    # strictly positive values.
    positive_image = image[image > 0]

    image_norm = LogNorm(
        vmin=positive_image.min(),
        vmax=positive_image.max(),
    )

    im = ax_img.imshow(
        image,
        origin="lower",
        cmap="gray",
        norm=image_norm,
    )

    # --------------------------------------------------------
    # Image colorbar
    # --------------------------------------------------------
    cbar_img = ax_img.figure.colorbar(
        im,
        ax=ax_img,
        fraction=0.046,
        pad=0.15,
        location="left",
    )

    cbar_img.set_label("Intensity [ADU]")

    # --------------------------------------------------------
    # Center = black star
    # --------------------------------------------------------
    ax_img.plot(
        cx,
        cy,
        marker="*",
        markersize=14,
        markerfacecolor="black",
        markeredgecolor="black",
        linestyle="None",
        zorder=20,
    )

    # --------------------------------------------------------
    # ST2 radius
    # --------------------------------------------------------
    ax_img.add_patch( Circle(
        (cx, cy),
        st2_rad,
        fill=False,
        edgecolor="black",
        linewidth=1.8,
        linestyle="--",
        zorder=10,
    ))

    # ax_img.add_patch(st2_circle)

    # --------------------------------------------------------
    # Coronagraph radius
    # --------------------------------------------------------
    ax_img.add_patch( Circle(
        (cx, cy),
        coro_rad,
        fill=False,
        edgecolor="black",
        linewidth=1.8,
        linestyle="--",
        zorder=10,
    ))

    # ax_img.add_patch(coro_circle)


    # ========================================================
    # APERTURES
    # ========================================================
    for j, (xx, yy) in enumerate(zip(x1, y1)):

        color = colors[j]

        # Selected side 
        ax_img.add_patch( Circle(
            (xx, yy),
            aperture_radius,
            fill=False,
            edgecolor=color,
            linewidth=3,
        ))

        # ax_img.add_patch(aperture)

        # Symmetric side
        ax_img.add_patch( Circle(
            (x2[j], y2[j]),
            aperture_radius,
            fill=False,
            edgecolor=color,
            linewidth=3.0,
            linestyle=":",
        ))

        # ax_img.add_patch(aperture_sym)

    ax_img.set_title(
        f"Sampling locations"
        #f"$\\theta={(ang for ang in angle if isinstance(angle, list) and angle is not None else angle):.1f}^\\circ$"
    )

    ax_img.set_xlabel("x [pixel]")
    ax_img.set_ylabel("y [pixel]")

    # ========================================================
    # SYMMETRIC INTENSITY GRAPH + LINEAR FIT
    # ========================================================
    
    fit_results = [] 
    linear_legend_handles = []
    quadratic_legend_handles = []
    for j in np.arange(size):
        r = radii[j % len(radii)]  # Use modulo to cycle through radii if angle is a list
        color = colors[j]
        x = np.asarray(left_int[:, j]) 
        y = np.asarray(symmetric_int[:, j])

        ax_int.scatter(
            x[::5],y[::5],
            color=color,
            s=8,
            alpha=0.5,
            linewidth=0,
        )

        # ---------------------------------------------------- 
        # # Remove invalid values before fitting 
        # # ---------------------------------------------------- 
        valid = ( np.isfinite(x) & np.isfinite(y) ) 
        x_fit = x[valid] 
        y_fit = y[valid]


        # --------------------------------------------------------
        # Linear and quadratic fits
        # --------------------------------------------------------
        
        if len(x_fit) >= 5 and np.ptp(x_fit) > 1e-12:

            # ====================================================
            # LINEAR FIT
            # ====================================================

            linear_coeffs = np.polyfit(
                x_fit,
                y_fit,
                1,
            )

            slope, intercept = linear_coeffs

            y_pred_linear = np.polyval(
                linear_coeffs,
                x_fit,
            )

            r_squared_linear = rsquared(
                y_fit,
                y_pred_linear,
            )
            # ====================================================
            # QUADRATIC FIT
            # ====================================================

            quadratic_coeffs = np.polyfit(
                x_fit,
                y_fit,
                2,
            )

            a, b, c = quadratic_coeffs

            y_pred_quadratic = np.polyval(
                quadratic_coeffs,
                x_fit,
            )

            r_squared_quadratic = rsquared(
                y_fit,
                y_pred_quadratic,
            )

            # ====================================================
            # PLOT BOTH FITS
            # ====================================================

            x_line = np.linspace(
                x_fit.min(),
                x_fit.max(),
                100,
            )

            # Linear
            y_line_linear = (
                slope * x_line
                + intercept
            )

            ax_int.plot(
                x_line,
                y_line_linear,
                color=color,
                linewidth=2,
                linestyle="-",
            )

            # Quadratic
            y_line_quadratic = (
                a * x_line**2
                + b * x_line
                + c
            )

            ax_int.plot(
                x_line,
                y_line_quadratic,
                color=color,
                linewidth=2,
                linestyle="--",
            )

        else:

            slope = np.nan
            intercept = np.nan
            r_squared_linear = np.nan

            a = np.nan
            b = np.nan
            c = np.nan
            r_squared_quadratic = np.nan


        # --------------------------------------------------------
        # Store fit
        # --------------------------------------------------------

        fit_results.append(
            {
                "radius": r,

                # Linear
                "slope": slope,
                "intercept": intercept,
                "r_squared_linear": r_squared_linear,

                # Quadratic
                "a": a,
                "b": b,
                "c": c,
                "r_squared_quadratic": r_squared_quadratic,
            }
        )


        # --------------------------------------------------------
        # Legend entries
        # --------------------------------------------------------

        if np.isfinite(slope):

            # Linear
            sign_linear = (
                "+"
                if intercept >= 0
                else "-"
            )

            linear_label = (
                f"r={r:.0f}: "
                f"y={slope:.3f}x "
                f"{sign_linear} "
                f"{abs(intercept):.2g}, "
                f"$R^2$={r_squared_linear:.3f}"
            )

            # Quadratic
            sign_b = "+" if b >= 0 else "-"
            sign_c = "+" if c >= 0 else "-"

            quadratic_label = (
                f"r={r:.0f}: "
                f"y={a:.3g}x² "
                f"{sign_b} {abs(b):.3g}x "
                f"{sign_c} {abs(c):.3g}, "
                f"$R^2$={r_squared_quadratic:.3f}"
            )

        else:

            linear_label = (
                f"r={r:.0f}: fit unavailable"
            )

            quadratic_label = (
                f"r={r:.0f}: fit unavailable"
            )


        # --------------------------------------------------------
        # Separate legend handles
        # --------------------------------------------------------

        linear_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                linestyle="-",
                label=linear_label,
            )
        )

        quadratic_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                linestyle="--",
                label=quadratic_label,
            )
        )

    #legend_handles.append( Line2D( [0], [0], color=color, linewidth=2, label=label, ) )
    
    ax_int.set_title(title)

    ax_int.set_xlabel("Left intensity [ADU]")
    ax_int.set_ylabel("180° symmetric intensity [ADU]")

    # --------------------------------------------------------
    # y = x reference line
    # --------------------------------------------------------
    xlim = ax_int.get_xlim()
    ylim = ax_int.get_ylim()

    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])

    ax_int.plot(
        [lo, hi],
        [lo, hi],
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax_int.set_xlim(0, hi * 0.8)
    ax_int.set_ylim(0, hi * 0.8)



    # ========================================================
    # Legend
    # ========================================================

    # ax_int.legend(
    #     handles=legend_handles,
    #     title="Linear fits",
    #     loc="best",
    #     fontsize=10,
    # )

    return linear_legend_handles, quadratic_legend_handles, fit_results



def plot_symmetric_comparison( int_cube, pred_cube, suptitle, angle=45.0, coro_rad=10, st2_rad=32, dr=5.0, crop=20, aperture_radius=1.5, r_aperture = 0.5
                              ): 
    """ 
    Plot Integrator Control and Predictive Control in one 2x2 figure. 
    
    Returns 
    ------- 
    fig : matplotlib.figure.Figure 
    axes : ndarray 2x2 array of axes. 
    fit_results : dict Linear-fit results for both datasets. 
    """ 
    # ======================================================== 
    # # Figure 
    # # ======================================================== 
    # 
    # fig, axes = plt.subplots( 2, 2, figsize=(15, 10), ) 

    fig = plt.figure(figsize=(20, 10))

    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1., 0.8, 1.7],
        wspace=0.15,
        hspace=0.3,
    )

    ax_img_int = fig.add_subplot(gs[0, 0])
    ax_int = fig.add_subplot(gs[0, 1])
    ax_legend_int = fig.add_subplot(gs[0, 2])

    ax_img_pred = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_legend_pred = fig.add_subplot(gs[1, 2])

    # ======================================================== 
    # # Integrator Control 
    # # ======================================================== 
    linear_legend_handles, quadratic_legend_handles, fit_int = plot_symmetric_intensities_single( 
            int_cube, ax_img_int, ax_int, 
            title="Integrator Control", 
            angle=angle, coro_rad=coro_rad, st2_rad=st2_rad, dr=dr, crop=crop, 
            aperture_radius=aperture_radius, 
            r_aperture = r_aperture
            ) 

    ax_legend_int.axis("off")

    legend_linear = ax_legend_int.legend(
        handles=linear_legend_handles,
        title="Linear fits",
        loc="upper left",
        fontsize=9,
        frameon=False,
    )

    legend_quadratic = ax_legend_int.legend(
        handles=quadratic_legend_handles,
        title="Quadratic fits",
        loc="upper right",
        fontsize=9,
        frameon=False,
    )

    ax_legend_int.add_artist(legend_linear)
    ax_legend_int.add_artist(legend_quadratic)
    # ======================================================== 
    # # Predictive Control 
    # # ======================================================== 
    linear_legend_handles_pred, quadratic_legend_handles_pred, fit_pred = plot_symmetric_intensities_single( 
            pred_cube, ax_img_pred, ax_pred, 
            title="Predictive Control", angle=angle, coro_rad=coro_rad, st2_rad=st2_rad, dr=dr, 
            crop=crop, aperture_radius=aperture_radius, 
            r_aperture = r_aperture)

    ax_legend_pred.axis("off")


    
    legend_linear_pred = ax_legend_pred.legend(
        handles=linear_legend_handles_pred,
        title="Linear fits",
        loc="upper left",
        fontsize=9,
        frameon=False,
    )

    legend_quadratic_pred = ax_legend_pred.legend(
        handles=quadratic_legend_handles_pred,
        title="Quadratic fits",
        loc="upper right",
        fontsize=9,
        frameon=False,
    )

    ax_legend_pred.add_artist(legend_linear_pred)
    ax_legend_pred.add_artist(legend_quadratic_pred)
    # ========================================================
    # Suptitle
    # ========================================================
    fig.suptitle(
        suptitle,
        fontweight = 'bold',
        fontsize=16,
    )

    fig.tight_layout(
        #rect=[0, 0, 1, 0.96]
    )

    # return fig, axes
    return ( 
        fig, 
        #axes, 
        { 
            "integrator": fit_int, 
            "predictive": fit_pred, 
            }, 
            )