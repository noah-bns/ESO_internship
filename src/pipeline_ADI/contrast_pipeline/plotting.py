
#org
import numpy as np
import re
from scipy import interpolate
import pandas as pd


#images
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


#scientific libraries
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.statistics import fpf_2_gaussian_sigma
import importlib
from . import functions_ADI
importlib.reload(functions_ADI)
from .functions_ADI import *







def save_grid_animation(contrast_grids, curves_output_path, dataset_name):

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

    ani.save(f"{(curves_output_path)}/GRID_{dataset_name}.gif", writer=PillowWriter(fps=2))



def plot_contrast_grid(
    contrast_grid_axis,
    colorbar_axis,
    contrast_grid,
    cmap = "YlGnBu",
    vmax=2, 
    vmin=7):

    c_bar_kargs = dict(
        orientation = "vertical",
        label = r"Confidence [$\sigma_{\mathcal{N}}$]")

    heat = sns.heatmap(
        contrast_grid,
        vmax=vmax, vmin=vmin,
        annot=True,
        cmap= cmap,
        fmt = '.1f',
        ax=contrast_grid_axis,
        cbar_ax=colorbar_axis,
        cbar_kws=c_bar_kargs)

    ylabels = ['{:.1f}'.format(float(x.get_text()))
               for x in heat.get_yticklabels()]
    _=heat.set_yticklabels(ylabels)
    xlabels = ['{:.1f}'.format(float(x.get_text()))
               for x in heat.get_xticklabels()]
    _=heat.set_xticklabels(xlabels)


def plot_contrast_curves_old(
    contrast_curves, 
    lim_mag_y,
    curves_output_path, 
    dataset_name,
    contrast_errors = None, 
    lim_x = None, 
    title = None,
    cmap = "magma",
    alpha = 0.7,
    ):

    # compute the overall best contrast curve
    PADI_values = contrast_curves.loc[:, ~contrast_curves.columns.str.contains("cADI", regex=True)]
    if len(PADI_values.columns) > 0:
        overall_best = np.min(PADI_values.values, axis=1)

        # get the error bars of the the overall best contrast curve
        best_idx = np.argmin(PADI_values.values, axis=1)

        # Find one color for each number of PCA components used
        color_map = plt.cm.get_cmap(cmap)   # seaborn-style colormap available in mpl
        colors = [color_map(int(i)) for i in np.round(np.linspace(0, 220, len(PADI_values.columns)))]

    if isinstance(contrast_curves.index, pd.MultiIndex):
        separations_arcsec = contrast_curves.index.get_level_values(0)
        separations_FWHM = contrast_curves.index.get_level_values(1)

        x = separations_FWHM        # bottom axis
        x_top = separations_arcsec  # top axis
        multi_index = True
    else:
        x = contrast_curves.index
        multi_index = False

    # 1.) Create Plot Layout
    fig = plt.figure(constrained_layout=False, figsize=(12, 8))
    gs0 = fig.add_gridspec(1, 1)
    axis_contrast_curves = fig.add_subplot(gs0[0, 0])


    # ---------------------- Create the Plot --------------------
    i = 0 # color picker
    for tmp_model in contrast_curves.columns:
        
        if 'cADI'.lower() in tmp_model.lower():
            num_components = 'cADI'
            color = 'red'
        else:
            num_components = int(tmp_model[5:8])
            color = colors[i]
            i+=1
        tmp_flux_ratios = contrast_curves.reset_index(
            level=0)[tmp_model].values


        axis_contrast_curves.plot(
            x,
            tmp_flux_ratios,
            color = color,
            alpha = alpha,
            label=num_components)

        if contrast_errors is not None:
            tmp_errors = contrast_errors.reset_index(
                level=0)[tmp_model].values
            
            axis_contrast_curves.fill_between(
                x,
                tmp_flux_ratios + tmp_errors,
                tmp_flux_ratios - tmp_errors,
                color = color,
                alpha=alpha/2)
            
            if len(PADI_values.columns) > 0:
                best_contrast_errors = contrast_errors.values[np.arange(len(best_idx)), best_idx]

                axis_contrast_curves.fill_between(
                    x,
                    overall_best + best_contrast_errors,
                    overall_best - best_contrast_errors,
                    color = 'blue',
                    alpha=alpha/2)

    axis_contrast_curves.set_yscale("log")
    # ------------ Plot the overall best -------------------------
    if len(PADI_values.columns) > 0:
        axis_contrast_curves.plot(
            x,
            overall_best,
            color = "blue",
            lw=3,
            ls="--",
            label="Best")

        best_pca = [
            int(re.search(r"PCA_(\d+)", col).group(1))
            for col in PADI_values.columns[best_idx]
        ]
        #save the overall best
        result = pd.DataFrame(
            {
                #"separation_FWHM": x,
                "best_contrast": overall_best,
                "best_PCA": best_pca,
            },
            index=PADI_values.index,
        )

    # ------------- Double axis and limits -----------------------
    if lim_x:
        lim_x = lim_x
    else:
        lim_x = (np.min(x) - 0.01, np.max(x) + 0.01)


    axis_contrast_curves_mag = axis_contrast_curves.twinx()
    axis_contrast_curves_mag.plot(
        x,
        flux_ratio2mag(tmp_flux_ratios),
        alpha=0.)
    axis_contrast_curves_mag.invert_yaxis()


    axis_contrast_curves.grid(which='both')
    axis_contrast_curves_mag.set_ylim(*lim_mag_y)
    axis_contrast_curves.set_ylim(
        (mag2flux_ratio(lim_mag_y[0])),
        (mag2flux_ratio(lim_mag_y[1])))

    axis_contrast_curves.set_xlim(*lim_x)
    axis_contrast_curves_mag.set_xlim(*lim_x)


    if multi_index:
        fwhm_to_arcsec = interpolate.interp1d(
            separations_FWHM,
            separations_arcsec,
            fill_value="extrapolate"
        )

        axis_contrast_curves_arcsec = axis_contrast_curves.twiny()
        axis_contrast_curves_arcsec.plot(
            separations_arcsec,
            tmp_flux_ratios,
            alpha=0,
        )

        axis_contrast_curves_arcsec.set_xlim(
            *fwhm_to_arcsec(lim_x)
        )
    # ----------- Labels and fontsizes --------------------------
    
        axis_contrast_curves_arcsec.set_xlabel(
            "Separation [arcsec]", size=16
        )
        axis_contrast_curves_arcsec.tick_params(
            axis='both', which='major', labelsize=14)
    
        axis_contrast_curves.set_xlabel(
            "Separation [FWHM]", size=16
        )
    else:
        axis_contrast_curves.set_xlabel(
            "Separation [FWHM]", size=16
        )

    axis_contrast_curves.set_ylabel(
        r"Planet-to-star flux ratio", size=16)
    axis_contrast_curves_mag.set_ylabel(
        r"$\Delta$ Magnitude", size=16)

    axis_contrast_curves.tick_params(
        axis='both', which='major', labelsize=14)

    axis_contrast_curves_mag.tick_params(
        axis='both', which='major', labelsize=14)

    # # ----------- Labels and fontsizes --------------------------
    if title:
        set_title = title
    else:
        set_title = r"$5 \sigma_{\mathcal{N}}$ Contrast Curves"
    axis_contrast_curves_mag.set_title(
        set_title,
        fontsize=18, fontweight="bold", y=1.05)

    # --------------------------- Legend -----------------------
    handles, labels = axis_contrast_curves.\
        get_legend_handles_labels()

    leg1 = fig.legend(handles, labels,
                    bbox_to_anchor=(0.12, -0.1),
                    fontsize=14,
                    title="# PCA components",
                    loc='lower left', ncol=8)

    _=plt.setp(leg1.get_title(),fontsize=14)
    plt.savefig(f"{(curves_output_path)}/Contrast_Curves_{dataset_name}.png", pad_inches=0.15)
    return result   

def plot_contrast_curves(
    contrast_curves, 
    lim_mag_y,
    curves_output_path, 
    dataset_name,
    contrast_errors = None, 
    lim_x = None, 
    title = None,
    cmap = "magma",
    alpha = 0.7,
    ):

    # compute the overall best contrast curve
    PADI_values = contrast_curves.loc[:, ~contrast_curves.columns.str.contains("cADI", regex=True)]
    if len(PADI_values.columns) > 0:
        overall_best = np.min(PADI_values.values, axis=1)

        # get the error bars of the the overall best contrast curve
        best_idx = np.argmin(PADI_values.values, axis=1)

        # Find one color for each number of PCA components used
        color_map = plt.cm.get_cmap(cmap)   # seaborn-style colormap available in mpl
        colors = [color_map(int(i)) for i in np.round(np.linspace(0, 220, len(PADI_values.columns)))]

    if isinstance(contrast_curves.index, pd.MultiIndex):
        separations_arcsec = contrast_curves.index.get_level_values(0)
        separations_FWHM = contrast_curves.index.get_level_values(1)

        x = separations_FWHM        # bottom axis
        x_top = separations_arcsec  # top axis
        multi_index = True
    else:
        x = contrast_curves.index
        multi_index = False

    # 1.) Create Plot Layout
    fig = plt.figure(constrained_layout=False, figsize=(12, 8))
    gs0 = fig.add_gridspec(1, 1)
    axis_contrast_curves = fig.add_subplot(gs0[0, 0])


    # ---------------------- Create the Plot --------------------
    i = 0 # color picker
    for tmp_model in contrast_curves.columns:
        
        if 'cADI'.lower() in tmp_model.lower():
            num_components = 'cADI'
            color = 'red'
        else:
            num_components = int(tmp_model[5:8])
            color = colors[i]
            i+=1
        tmp_flux_ratios = contrast_curves.reset_index(
            level=0)[tmp_model].values


        axis_contrast_curves.plot(
            x,
            tmp_flux_ratios,
            color = color,
            alpha = alpha,
            label=num_components)

        if contrast_errors is not None:
            tmp_errors = contrast_errors.reset_index(
                level=0)[tmp_model].values
            
            axis_contrast_curves.fill_between(
                x,
                tmp_flux_ratios + tmp_errors,
                tmp_flux_ratios - tmp_errors,
                color = color,
                alpha=alpha/2)
            
            if len(PADI_values.columns) > 0:
                best_contrast_errors = contrast_errors.values[np.arange(len(best_idx)), best_idx]

                axis_contrast_curves.fill_between(
                    x,
                    overall_best + best_contrast_errors,
                    overall_best - best_contrast_errors,
                    color = 'blue',
                    alpha=alpha/2)

    axis_contrast_curves.set_yscale("log")
    # ------------ Plot the overall best -------------------------
    if len(PADI_values.columns) > 0:
        axis_contrast_curves.plot(
            x,
            overall_best,
            color = "blue",
            lw=3,
            ls="--",
            label="Best")

        best_pca = [
            int(re.search(r"PCA_(\d+)", col).group(1))
            for col in PADI_values.columns[best_idx]
        ]
        #save the overall best
        result = pd.DataFrame(
            {
                #"separation_FWHM": x,
                "best_contrast": overall_best,
                "best_PCA": best_pca,
            },
            index=PADI_values.index,
        )

    # ------------- Double axis and limits -----------------------
    if lim_x:
        lim_x = lim_x
    else:
        lim_x = (np.min(x) - 0.01, np.max(x) + 0.01)


    axis_contrast_curves_mag = axis_contrast_curves.twinx()
    axis_contrast_curves_mag.plot(
        x,
        flux_ratio2mag(tmp_flux_ratios),
        alpha=0.)
    axis_contrast_curves_mag.invert_yaxis()


    axis_contrast_curves.grid(which='both')
    axis_contrast_curves_mag.set_ylim(*lim_mag_y)
    axis_contrast_curves.set_ylim(
        (mag2flux_ratio(lim_mag_y[0])),
        (mag2flux_ratio(lim_mag_y[1])))

    axis_contrast_curves.set_xlim(*lim_x)
    axis_contrast_curves_mag.set_xlim(*lim_x)


    if multi_index:
        fwhm_to_arcsec = interpolate.interp1d(
            separations_FWHM,
            separations_arcsec,
            fill_value="extrapolate"
        )

        axis_contrast_curves_arcsec = axis_contrast_curves.twiny()
        axis_contrast_curves_arcsec.plot(
            separations_arcsec,
            tmp_flux_ratios,
            alpha=0,
        )

        axis_contrast_curves_arcsec.set_xlim(
            *fwhm_to_arcsec(lim_x)
        )
    # ----------- Labels and fontsizes --------------------------
    
        axis_contrast_curves_arcsec.set_xlabel(
            "Separation [arcsec]", size=16
        )
        axis_contrast_curves_arcsec.tick_params(
            axis='both', which='major', labelsize=14)
    
        axis_contrast_curves.set_xlabel(
            "Separation [FWHM]", size=16
        )
    else:
        axis_contrast_curves.set_xlabel(
            "Separation [FWHM]", size=16
        )

    axis_contrast_curves.set_ylabel(
        r"Planet-to-star flux ratio", size=16)
    axis_contrast_curves_mag.set_ylabel(
        r"$\Delta$ Magnitude", size=16)

    axis_contrast_curves.tick_params(
        axis='both', which='major', labelsize=14)

    axis_contrast_curves_mag.tick_params(
        axis='both', which='major', labelsize=14)

    # # ----------- Labels and fontsizes --------------------------
    if title:
        set_title = title
    else:
        set_title = r"$5 \sigma_{\mathcal{N}}$ Contrast Curves"
    axis_contrast_curves_mag.set_title(
        set_title,
        fontsize=18, fontweight="bold", y=1.05)

    # --------------------------- Legend -----------------------
    handles, labels = axis_contrast_curves.\
        get_legend_handles_labels()

    leg_loc_y = (0.005*len(PADI_values.columns))
    leg1 = fig.legend(handles, labels,
                    bbox_to_anchor=(0.12, -leg_loc_y),
                    fontsize=14,
                    title="# PCA components",
                    loc='lower left', ncol=8)

    _=plt.setp(leg1.get_title(),fontsize=14)
    plt.savefig(f"{(curves_output_path)}/Contrast_Curves_{dataset_name}.png",
        bbox_inches="tight",
        bbox_extra_artists=(leg1,),
    )
    return result   


def plot_overall_best(grid, curves_output_path, dataset_name, pca = False):

    fig = plt.figure(figsize=(8, 4))

    gs0 = fig.add_gridspec(1, 1)
    gs0.update(wspace=0.0, hspace=0.2)
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec = gs0[0],
        wspace=0.05, width_ratios=[1, 0.03])

    # All axis we need
    contrast_ax = fig.add_subplot(gs1[0])
    colorbar_ax = fig.add_subplot(gs1[1])
    if pca == False:
        #grid = grid.map(fpf_2_gaussian_sigma)
        title = 'Overall Best Performance'
    else:
        title = 'Overall Best - PCA component'

    # Plot the contrast grid
    plot_contrast_grid(
        contrast_grid_axis=contrast_ax,
        colorbar_axis=colorbar_ax,
        contrast_grid=grid, 
        cmap = 'YlGn' if pca == True else "YlGnBu",
        vmax = np.max(grid) if pca == True else 2,
        )

    contrast_ax.set_ylabel(
        "Contrast - $c = f_p / f_*$ - [mag]", size=14)
    contrast_ax.set_xlabel(
        r"Separation [FWHM]", size=14)
    contrast_ax.set_title(
        "Contrast Grid: " + title + '\n' + dataset_name.replace('_', ' '),
        fontsize=16,
        fontweight="bold",
        y=1.03)

    contrast_ax.tick_params(
        axis='both',
        which='major',
        labelsize=12)

    # Save the figure
    fig.patch.set_facecolor('white')

    plt.savefig(f"{(curves_output_path)}/GRID_{title.replace(' ', '_')}_{dataset_name}.png", pad_inches=0.2)


def plot_all_best_old(curves_output_path):
    with pd.HDFStore(f"{curves_output_path}/overall_best.h5") as store:
        fig, ax = plt.subplots(figsize=(12, 7))
        gs0 = fig.add_gridspec(1, 1)
        color_map = plt.cm.get_cmap('magma')   # seaborn-style colormap available in mpl
        colors = [color_map(int(i)) for i in np.round(np.linspace(10, 150, len(store.keys())))]



        for i, key in enumerate(store.keys()):
            df = store[key]

            ax.semilogy(
                df.index,
                df["best_contrast"],
                lw=2,
                color = colors[i],
                label=key.strip("/").replace('_', ' ').replace('windspeed', 'ws'),
            )

            sc = ax.scatter(
                df.index,
                df["best_contrast"],
                c=df["best_PCA"],
                cmap="YlGn",
                s=100,
                zorder=5,
                linewidth = 0.5,
                edgecolors= colors[i],
            )

        leg1 = fig.legend(
            fontsize=14,
            loc="lower left",
            bbox_to_anchor=(0.1, -(0.05 * len(store.keys()))),
            ncol=2,
            title="Experiment",
            title_fontsize=16, 
        )
        
    # Colorbar
    cbar = fig.colorbar(
        sc,
        ax=ax,               # or axis_contrast_curves
        pad=0.1,            # distance from the axes
    )

    cbar.set_label(
        "Best PCA",
        fontsize=16,
    )

    cbar.ax.tick_params(labelsize=14)
    ax.set_title(
            'Overall Best Contrast Curve',
            fontsize=18, fontweight="bold", y=1.05)

    ax.set_xlabel("Separation [FWHM]", fontsize=16)
    ax.set_ylabel("Planet-to-star flux ratio", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(which='both')
    ax_mag = ax.twinx()
    ax_mag.set_ylabel(r"$\Delta$ Magnitude", fontsize=16)

    ax_mag.tick_params(axis="both", which="major", labelsize=14)

    plt.savefig(f"{(curves_output_path)}/Overall_Best_Contrast.png",
            bbox_inches="tight",
            bbox_extra_artists=(leg1,),
        )


def plot_all_best2(curves_output_path):
    with pd.HDFStore(f"{curves_output_path}/overall_best.h5") as store:
        fig, ax = plt.subplots(figsize=(12, 7))
        gs0 = fig.add_gridspec(1, 1)
        # color_map = plt.cm.get_cmap('inferno')   # seaborn-style colormap available in mpl
        # colors = [color_map(int(i)) for i in np.round(np.linspace(0, 250, len(store.keys())))]
        color_map = plt.colormaps["tab20c"]
        colors = color_map(np.linspace(0, 1, len(store)))

        reds = plt.colormaps["Reds"]
        blues = plt.colormaps["Blues"]

        int_keys = [k for k in store.keys() if k.endswith("int")]
        pred_keys = [k for k in store.keys() if k.endswith("pred")]

        red_colors = dict(zip(int_keys, reds(np.linspace(0.2, 1, len(int_keys)))))
        blue_colors = dict(zip(pred_keys, blues(np.linspace(0.2, 1, len(pred_keys)))))

        # Then inside the loop:

        ordered_keys = int_keys + pred_keys

        for key in ordered_keys:
            df = store[key]

            line_color = (
                red_colors[key] if key.endswith("int")
                else blue_colors[key]
            )           
            ax.semilogy(
                df.index,
                df["best_contrast"],
                lw=2,
                color = line_color, #colors[i],
                label=key.strip("/").replace('_', ' ').replace('windspeed', 'ws'),
            )

            sc = ax.scatter(
                df.index,
                df["best_contrast"],
                c=df["best_PCA"],
                cmap="YlGn",
                s=100,
                zorder=5,
                linewidth = 0.5,
                edgecolors= line_color,
            )

        leg1 = fig.legend(
            fontsize=14,
            loc="lower left",
            bbox_to_anchor=(0.1, -(0.02 * len(store.keys()))),
            ncol=3,
            title="Experiment",
            title_fontsize=16, 
        )
        
    # Colorbar
    cbar = fig.colorbar(
        sc,
        ax=ax,               # or axis_contrast_curves
        pad=0.1,            # distance from the axes
    )

    cbar.set_label(
        "Best PCA",
        fontsize=16,
    )

    cbar.ax.tick_params(labelsize=14)
    ax.set_title(
            'Overall Best Contrast Curves',
            fontsize=18, fontweight="bold", y=1.05)

    ax.set_xlabel("Separation [FWHM]", fontsize=16)
    ax.set_ylabel("Planet-to-star flux ratio", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(which='both')
    ax_mag = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax_mag.set_ylim(flux_ratio2mag(ymin), mag2flux_ratio(ymax))
    ax_mag.set_ylabel(r"$\Delta$ Magnitude", fontsize=16)
    # ax_mag = ax.secondary_yaxis(
    #     "right",
    #     functions=(flux_ratio2mag, mag2flux_ratio)  # (forward, inverse)
    # )
    ax_mag.tick_params(axis="both", which="major", labelsize=14)

    plt.savefig(f"{(curves_output_path)}/Overall_Best_Contrast.png",
            bbox_inches="tight",
            bbox_extra_artists=(leg1,),
        )

def plot_all_best(curves_output_path):
    with pd.HDFStore(f"{curves_output_path}/overall_best.h5") as store:
        fig, ax = plt.subplots(figsize=(12, 7))
        gs0 = fig.add_gridspec(1, 1)
        # color_map = plt.cm.get_cmap('inferno')   # seaborn-style colormap available in mpl
        # colors = [color_map(int(i)) for i in np.round(np.linspace(0, 250, len(store.keys())))]
        color_map = plt.colormaps["tab20c"]
        colors = color_map(np.linspace(0, 1, len(store)))
        seq = 'managua'
        reds = plt.colormaps[seq]
        blues = plt.colormaps[seq]

        int_keys = [k for k in store.keys() if k.endswith("int")]
        pred_keys = [k for k in store.keys() if k.endswith("pred")]

        red_colors = dict(zip(int_keys, reds(np.linspace(0.2, 1, len(int_keys)))))
        blue_colors = dict(zip(pred_keys, blues(np.linspace(0.2, 1, len(pred_keys)))))

        # Then inside the loop:

        ordered_keys = pred_keys + int_keys

        df_0 = store[ordered_keys[0]]
        ax.semilogy(
            df_0.index,
            df_0["best_contrast"],
            lw=1.2,
            color = 'black', #colors[i],
            ls = '-',
            label= 'Predictive Control           ',
        )
        ax.semilogy(
            df_0.index,
            df_0["best_contrast"],
            lw=1.5,
            color = 'black', #colors[i],
            ls = '--',
            label= 'Integrator Control             ',
        )

        for key in ordered_keys:
            df = store[key]

            line_color = (
                red_colors[key] if key.endswith("int")
                else blue_colors[key]
            )

            mark = (
                '--' if key.endswith("int")
                else '-'
            )    

            label =   key.strip("/").replace('_', ' ').replace('windspeed', ' ws').replace('r0', 'e-1 r0,').replace('pred', '')  
            ax.semilogy(
                df.index,
                df["best_contrast"],
                lw=3,
                color = line_color, #colors[i],
                ls = mark,
                label= label if key in pred_keys else None,
            )

            sc = ax.scatter(
                df.index,
                df["best_contrast"],
                c=df["best_PCA"],
                cmap=plt.colormaps["YlGn"],
                s=100,
                zorder=5,
                linewidth = 0.2,
                edgecolors= line_color, #'grey', #colors[i],
            )

        leg1 = fig.legend(
            fontsize=14,
            loc="lower left",
            bbox_to_anchor=(0.12, -(0.018 * len(store.keys()))),
            ncol=2,
            title="Experiment",
            title_fontsize=16, 
        )
        
    # Colorbar
    cbar = fig.colorbar(
        sc,
        ax=ax,               # or axis_contrast_curves
        pad=0.1,            # distance from the axes
    )

    cbar.set_label(
        "Best PCA",
        fontsize=16,
    )

    cbar.ax.tick_params(labelsize=14)
    ax.set_title(
            'Overall Best Contrast Curves',
            fontsize=18, fontweight="bold", y=1.05)

    ax.set_xlabel("Separation [FWHM]", fontsize=16)
    ax.set_ylabel("Planet-to-star flux ratio", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(which='both')
    ax_mag = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax_mag.set_ylim(flux_ratio2mag(ymin), flux_ratio2mag(ymax))
    ax_mag.set_ylabel(r"$\Delta$ Magnitude", fontsize=16)
    # ax_mag = ax.secondary_yaxis(
    #     "right",
    #     functions=(flux_ratio2mag, mag2flux_ratio)  # (forward, inverse)
    # )
    ax_mag.tick_params(axis="both", which="major", labelsize=14)

    plt.savefig(f"{(curves_output_path)}/Overall_Best_Contrast.png",
            bbox_inches="tight",
            bbox_extra_artists=(leg1,),
        )

curves_output_path = '/home/aosimul/noah/src/pipeline_ADI/results/contrast'
plot_all_best(curves_output_path)