import os
import string
from itertools import combinations
from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import scipy
from PIL import Image, ImageDraw
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pdf2image import convert_from_path
from skimage.color import rgb2lab, lab2rgb

import constants as c
import idr_utils as utils
from data_fitting import plot_full_data_response_vs_fit

# plt.switch_backend('Qt5Agg')
plt.style.use('comdepri.mplstyle')


def close_all():
    plt.close('all')


# plotting utility functions
def get_normed_color_mix(c1, c2, values):
    """
    Get perceptually uniform mixture of 2 colors based on the normed values
    :param c1:
    :param c2:
    :param values:
    :return:
    """
    values = np.array(values, dtype=np.float64)
    c1 = rgb2lab(np.array(colors.to_rgb(c1)))
    c2 = rgb2lab(np.array(colors.to_rgb(c2)))
    c_range = values - np.min(values)
    c_range /= np.max(c_range)
    color_arr = [colors.to_hex(lab2rgb((1 - alpha) * c1 + alpha * c2)) for alpha in c_range]
    return color_arr


def get_color_mix(c1, c2, num_points):
    """
    Get a list of colors that are gradients between 2 colors
    :param c1:
    :param c2:
    :param num_points:
    :return:
    """
    c1 = rgb2lab(np.array(colors.to_rgb(c1)))
    c2 = rgb2lab(np.array(colors.to_rgb(c2)))
    c_range = np.linspace(0, 1, num_points)
    color_arr = [colors.to_hex(lab2rgb((1 - alpha) * c1 + alpha * c2)) for alpha in c_range]
    return color_arr


def get_fig_size(nrow, ncol):
    """
    Returns the figures size for a figure with nrow rows and ncol columns
    """
    return c.SUBPLOT_SIZE * ncol, c.SUBPLOT_SIZE * nrow


def set_ax_labels(ax: plt.Axes, title: str, x: str, y: str, title_size=c.AX_TITLE_SIZE, ax_label_size=c.AX_LABEL_SIZE,
                  xlabelpad=c.X_LABELPAD, ylabelpad=c.Y_LABELPAD):
    """
    Sets the labels of a plt.Axes.
    :param ax: The axes to set labels for
    :param title: title of the axes
    :param x: x label of the axes
    :param y: y label of the axes
    :param title_size: the fontsize of the title
    :param ax_label_size: the fontsize of axis labels
    :param xlabelpad: padding for x label between the label and the ticks
    :param ylabelpad: padding for y label between the label and the ticks
    """
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(x, fontsize=ax_label_size, labelpad=xlabelpad)
    ax.set_ylabel(y, fontsize=ax_label_size, labelpad=ylabelpad)


def add_error_bars(ax, x, y, low_bars, high_bars, width, color='k'):
    """
    Adds error bars to an axis
    :param ax: the axis
    :param x: x values of the plotted values
    :param y: y values of the plotted values
    :param low_bars: y values for the low end of the error bars
    :param high_bars: y values for the high end of the error bars
    :param width: the width of the horizontal lines at the ends of the error bars
    :param color: the color of the error bars
    """
    x, y, low_bars, high_bars = list(map(np.array, [x, y, low_bars, high_bars]))
    ax.vlines(x, low_bars, high_bars, colors=color)
    ax.hlines(low_bars, x - width, x + width, colors=color)
    ax.hlines(high_bars, x - width, x + width, colors=color)
    ax.set_ylim(min(y.min(), -np.abs(y).max() / 10),
                max(y.max(), np.abs(y).max() / 10))
    ax.set_xticks(x)


def plot_shift_func(x, y, title):
    """
    Plot a shift function of x-y
    :param x: First vector
    :param y: Second vector
    :param title: title of the plot ("shift function" is added after it)
    :return: the figure on which the shift function is plotted
    """
    deciles = np.arange(0.1, 1, 0.1)
    decile_diff_mean, diff_low_ci, diff_high_ci, decile_diff = utils.shift_function(x, y, True)
    fig: plt.Figure = plt.figure(figsize=get_fig_size(1, 1.3))
    ax: plt.Axes = fig.subplots()
    ax.scatter(deciles, decile_diff_mean, c='k')
    add_error_bars(ax, deciles, decile_diff, diff_low_ci, diff_high_ci, 0.02)
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyles=":", colors="k")
    set_ax_labels(ax, title + " shift function",
                  "decile", "decile difference", 0.55 * c.AX_TITLE_SIZE, 0.55 * c.AX_LABEL_SIZE)
    return fig


def set_labels(axes: [plt.Axes, list, np.array],
               titles: [str, list, np.array],
               x: [str, list, np.array] = "x",
               y: [str, list, np.array] = "y", title_size=None, ax_label_size=None) -> None:
    """
    Sets the labels of given axes. Can handle lists of axes, lists of labels.
    :param axes: axis or list of axes to set labels for
    :param titles: title or list of titles for the axes
    :param x: x label or list of x labels for the axes
    :param y: y label or list of y labels for the axes
    :param title_size: size of the titles for all axes
    :param ax_label_size: size of the
    """
    if not title_size:
        title_size = c.AX_TITLE_SIZE
    if not ax_label_size:
        ax_label_size = c.AX_LABEL_SIZE
    iterable_title = False
    iterable_x = False
    iterable_y = False
    try:
        iter(axes)
        iterable_ax = True
        try:
            iter(titles)
            if isinstance(titles, str):
                raise TypeError
            iterable_title = True
        except TypeError:
            iterable_title = False
        try:
            iter(x)
            if isinstance(x, str):
                raise TypeError
            iterable_x = True
        except TypeError:
            iterable_x = False
        try:
            iter(y)
            if isinstance(y, str):
                raise TypeError
            iterable_y = True
        except TypeError:
            iterable_y = False
    except TypeError:
        iterable_ax = False
    if iterable_ax:
        for i, ax in enumerate(axes):
            set_ax_labels(ax, titles[i] if iterable_title else titles, x[i] if iterable_x else x,
                          y[i] if iterable_y else y, title_size, ax_label_size)
    else:
        set_ax_labels(axes, titles, x, y, title_size, ax_label_size)


def make_log_scale(ax: [plt.Axes, List[plt.Axes]], x=True, y=True):
    """
    Makes a plot logarithmically scaled
    :param ax: axis or list of axes to transform to log scale
    :param x: bool, whether to transform x-axis to log scale
    :param y: bool, whether to transform y-axis to log scale
    """
    try:
        iter(ax)
        for axis in ax:
            axis: plt.Axes
            axis.grid(False)
            if x:
                axis.set_xscale("log")
            if y:
                axis.set_yscale("log")
    except TypeError:
        ax.grid(False)
        if x:
            ax.set_xscale("log")
        if y:
            ax.set_yscale("log")


def savefig(fig: plt.Figure, save_name: str, shift_x=-0.1, shift_y=1.01, ignore: [None, list] = None, si=False,
            tight=True, numbering_size=40, figure_coord_labels=None, width=None, dpi=300) -> None:
    """
    Saves a figure to the default figure directory. Adds letter numbering to plots.
    :param fig: Figure to save
    :param save_name: Name of the saved figure
    :param shift_x: how much (in axes coordinates) to shift the lettering from 0
    :param shift_y: how much (in axes coordinates) to shift the lettering from 0
    :param ignore: list of axis to ignore when adding lettering
    :param si: whether the plot should be saved to the supplementary figures directory
    :param tight: whether tight_layout should be applied to the figure
    :param numbering_size: fontsize of the lettering
    """
    if ignore is None:
        ignore = []
    if not save_name.lower().endswith(".pdf") and not save_name.lower().endswith(".tiff"):
        save_name += ".pdf"
    if tight:
        fig.tight_layout()
    if not (figure_coord_labels is None):
        for i, coord in enumerate(figure_coord_labels):
            fig.text(coord[0], coord[1], string.ascii_uppercase[i], size=numbering_size, weight='bold', ha='center')
    elif len(fig.axes) - len(ignore) > 1:
        for i, ax in enumerate(fig.axes):
            if ax in ignore or ax.get_label() == '<colorbar>':
                continue
            ax.text(shift_x, shift_y, string.ascii_uppercase[i], transform=ax.transAxes,
                    size=numbering_size, weight='bold')
    save_path = os.path.join(c.SI_FIG_DIR if si else c.FIG_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if width is not None:
        width = 0.393701 * width  # to inches
        w, h = fig.get_size_inches()
        cur_ratio = h / w
        fig.set_size_inches(width, width * cur_ratio)
    fig.savefig(save_path, pad_inches=0.15, dpi=dpi)


def get_fig_with_centered_last_row(n_figs, n_rows=None, n_cols=None):
    """
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.

    It returns a grid where the plots
    in the **last** row are centered.

    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    if n_rows is not None and n_cols is not None:
        assert n_figs < n_rows * n_cols, "No empty subplots, use normal plt.subplots() instead"
    elif n_rows is not None:
        n_cols = n_figs // n_rows + 1 * (n_figs % n_rows > 0)
    elif n_cols is not None:
        n_rows = n_figs // n_cols + 1 * (n_figs % n_cols > 0)
    else:
        n_rows = int(np.sqrt(n_figs))
        n_cols = n_rows - 1 + n_figs - n_rows ** 2

    fig = plt.figure(figsize=get_fig_size(n_rows, n_cols))
    axs = []

    m = n_figs % n_cols
    m = range(1, n_cols + 1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(n_rows, m * n_cols)

    for i in range(0, n_figs):
        row = i // n_cols
        col = i % n_cols

        if row == n_rows - 1:  # center only last row
            off = int(m * (n_cols - n_figs % n_cols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m * col + off: m * (col + 1) + off])
        axs.append(ax)

    return fig, axs


def get_3_axes_with_3rd_centered():
    """
    Get 3 subplots with the last one centered
    :return: figure, axes list
    """
    fig: plt.Figure = plt.figure(figsize=get_fig_size(2.5, 2.5))
    axes = np.empty(3, plt.Axes)
    axes[0] = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    axes[1] = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    axes[2] = plt.subplot2grid((4, 4), (2, 1), colspan=2, rowspan=2)
    return fig, axes


def get_3_axes_with_1st_full():
    """
    Get 3 subplots with the last one centered
    :return: figure, axes list
    """
    fig: plt.Figure = plt.figure(figsize=get_fig_size(2, 2.5))
    axes = np.empty(3, plt.Axes)
    axes[0] = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
    axes[1] = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)
    axes[2] = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)
    return fig, axes


def plot_hist_with_stat(statistic, hist, title="", xlabel="Statistic", ylabel="Frequency", p=None, hist_color=None,
                        line_color=None, bins=None):
    """
    Plots a histogram
    :param statistic: value of the statistic to mark
    :param hist: 1D array with histogram values
    :param title: title of the figure
    :param xlabel: x-label of the figure
    :param ylabel: y-label of the figure
    :param p: optional, pvalue of th estatistic
    :param hist_color: optional, color of the histogram
    :param line_color: optional, color of the statistic line
    :return: the figure plotted on
    """
    fig, ax = plt.subplots()
    ax: plt.Axes
    if bins is None:
        bins = max(50, hist.size // 100)
    if hist_color is not None:
        ax.hist(hist, bins=bins, color=hist_color)
    else:
        ax.hist(hist, bins=bins)
    if hist_color is not None:
        l = ax.axvline(statistic, color=line_color, linestyle=":", linewidth=1.3)
    else:
        l = ax.axvline(statistic, color='k', linestyle=":", linewidth=1.3)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if p is not None:
        ax.legend([l], [f"p {'=' if p > 1e-4 else '<'} {utils.num2latex(p)}"])
    return fig


def select_significance_text(pval):
    """
    Transforms p value to significance marking
    :param pval: p value
    :return: significance marking
    """
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return '*'
    return "n.s"


def plot_boxplots(boxplot_value_list, x_labels, group_labels, title, xlabel, ylabel, colors=None, p_val_test_func=None,
                  fig=None, ax=None, save=True):
    """
    Plots boxplots of several groups under different conditions/values side by side and saves the plot.
    :param boxplot_value_list: 2D array with first axis corresponding to different groups
    :param x_labels: list of strings, labels for the x-axis ticks
    :param group_labels: list of strings, names of the different groups.
    :param title: Title of the figure
    :param xlabel: x-label
    :param ylabel: y-label
    :param colors: colors for the different groups. If None, uses default color cycle
    :param p_val_test_func: optional. Function that calculates p value of difference statistic between 2 groups.
                            If passed, adds significance markers between groups
    """
    boxplot_value_list = list(map(np.array, boxplot_value_list))
    n_groups = len(group_labels)
    n_xticks = len(x_labels)
    # Assertions
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    if len(colors) < n_groups:
        raise AssertionError("If you did not provide colors, the default color cycle has less colors than groups!\n"
                             "If you provided colors, you did not provide enough!\n"
                             "Please provide a list of color names with at least as many colors as groups to plot.")
    if len(boxplot_value_list) != n_groups:
        raise AssertionError(
            f"len(boxplot_value_list) should be the same as len(group_labels), "
            f"yet it is {len(boxplot_value_list)} and {n_groups}")
    if len(boxplot_value_list[0].shape) > 1:
        if boxplot_value_list[0].shape[1] != n_xticks:
            raise AssertionError(
                f"Column number of boxplot_value_list should be the same as len(x_labels), "
                f"yet it is {boxplot_value_list[0].shape[1]} and {n_xticks}")

    if fig is None:
        fig = plt.figure(figsize=get_fig_size(1, 1.3))
    if ax is None:
        ax: plt.Axes = fig.subplots()
    boxplot_center_positions = np.arange(1, n_xticks + 1) * n_groups
    offsets = np.arange(1, n_groups + 1) * 0.6
    offsets -= offsets.mean()
    bp_list = []
    for i in range(n_groups):
        bp = ax.boxplot(boxplot_value_list[i], patch_artist=True,
                        positions=np.round(boxplot_center_positions + offsets[i], 3),
                        showfliers=False,
                        capprops={"color": colors[i], "alpha": 0.7},
                        notch=True,
                        medianprops={"color": "black"},
                        whiskerprops=dict(color=colors[i], alpha=0.7),
                        boxprops=dict(facecolor=colors[i], color="black"))
        bp_list.append(bp)
    ax.set_xticks(boxplot_center_positions)
    if p_val_test_func is not None:
        idx_pairs = list(combinations(range(n_groups), 2))
        idx_pairs.sort(key=lambda x: x[1] - x[0])
        dy = 0.06 * ax.get_ylim()[1]
        for idx_pair in idx_pairs:
            a, b = boxplot_value_list[idx_pair[0]], boxplot_value_list[idx_pair[1]]
            p_vals = p_val_test_func(a, b)
            texts = [select_significance_text(pval) for pval in p_vals]
            text_heights = np.maximum(np.squeeze(np.quantile(a, [.8], 0)),
                                      np.squeeze(np.quantile(b, [.8], 0))) + (idx_pair[1] - idx_pair[0] - 1) * dy
            ax.hlines(text_heights, boxplot_center_positions + offsets[idx_pair[0]] + 0.05,
                      boxplot_center_positions + offsets[idx_pair[1]] - 0.05, colors="black")
            vline_height = text_heights[-2] * .02
            ax.vlines(boxplot_center_positions + offsets[idx_pair[0]] + 0.05, text_heights + vline_height,
                      text_heights - vline_height,
                      colors="black")
            ax.vlines(boxplot_center_positions + offsets[idx_pair[1]] - 0.05, text_heights + vline_height,
                      text_heights - vline_height,
                      colors="black")
            for i, height in enumerate(text_heights):
                ax.text(boxplot_center_positions[i], height, texts[i], horizontalalignment='center')
    ax.legend([bp["boxes"][0] for bp in bp_list], group_labels)
    ax.set_xticklabels(x_labels)
    set_ax_labels(ax, title, xlabel, ylabel)
    if save:
        savefig(fig, title, ignore=[ax], si=True)


# =======================================================
# ============ simulation plotting functions ============
# =======================================================

def plot_individual_resps_with_mean(ax, x, y_mat, color):
    ax.plot(x, y_mat.T, color="gray", alpha=0.3,
            label="Individual", zorder=1)
    ax.plot(x, y_mat.mean(0), color=color,
            label=r"Population", linewidth=2, zorder=2, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[-1]]
    labels = [labels[0], labels[-1]]
    ax.legend(handles, labels, loc='upper left', fontsize=25)


def plot_population_response(s, asd_resp, nt_resp) -> plt.Figure:
    fig_population_resp = plt.figure(figsize=get_fig_size(1, 2))
    axes = fig_population_resp.subplots(1, 2)
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.01)
    # plot population response
    plot_individual_resps_with_mean(axes[0], s, nt_resp, c.NT_COLOR)
    plot_individual_resps_with_mean(axes[1], s, asd_resp, c.ASD_COLOR)
    set_labels(axes, "", "Signal Level", r"Neural Gain")
    for ax in axes:  # type: plt.Axes
        ax.tick_params(axis='both', which='major', labelsize=23)
    return fig_population_resp


def plot_variance_over_signal_range(signal, nt_variance, asd_variance, ei_variance):
    fig = plt.figure(figsize=get_fig_size(1, 1.2))
    ax: plt.Axes = fig.subplots()
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.05)
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    set_ax_labels(ax,
                  "",
                  "Signal level", "Population response\nvariance", ax_label_size=27)
    ax.scatter(signal, nt_variance, s=2, alpha=0.5, c=c.NT_COLOR, label="NDR")
    ax.text(0.22, 6e-4, "NDR", color=c.NT_COLOR, size=25, weight="bold")
    ax.scatter(signal, asd_variance, s=2, alpha=0.5, c=c.ASD_COLOR, label="IDR")
    ax.text(0.195, 2.9e-4, "IDR", color=c.ASD_COLOR, size=25, weight="bold")
    ax.scatter(signal, ei_variance, s=2, alpha=0.5, c=c.EI_COLOR, label="E/I")
    ax.text(0.65, 7.5e-4, "E/I", color=c.EI_COLOR, size=25, weight="bold")
    return fig


def plot_variance_over_signal_range_rosenberg(signal, nt_variance, asd_variance, ax=None):
    if ax is None:
        fig = plt.figure(figsize=get_fig_size(1, 1.2))
        ax: plt.Axes = fig.subplots()
    else:
        fig = ax.get_figure()
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.05)
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    set_ax_labels(ax,
                  "",
                  "Signal level", "Population response\nvariance", ax_label_size=27)
    ax.scatter(signal, nt_variance, s=2, alpha=0.5, c=c.NT_COLOR, label="NDR")
    ax.text(0.025, -.25e-4, "Full inhibition", color=c.NT_COLOR, size=25, weight="bold")
    ax.scatter(signal, asd_variance, s=2, alpha=0.5, c=c.EI_COLOR, label="IDR")
    ax.text(0.15, 6e-4, "Decreased inhibition", color=c.EI_COLOR, size=25, weight="bold")
    return fig


def plot_km_tracking(ax: plt.Axes, km, time):
    ax.plot(time, km[:, :, 0].mean(0), color=c.NT_COLOR, zorder=1, linewidth=3, label="NDR")
    ax.fill_between(time, km[:, :, 0].mean(0) + km[:, :, 0].std(0), km[:, :, 0].mean(0) - km[:, :, 0].std(0), alpha=0.3,
                    color=c.NT_COLOR)

    ax.plot(time, km[:, :, 2].mean(0), color=c.ASD_COLOR, zorder=2, linewidth=3, label="IDR")
    ax.fill_between(time, km[:, :, 2].mean(0) + km[:, :, 2].std(0), km[:, :, 2].mean(0) - km[:, :, 2].std(0), alpha=0.3,
                    color=c.ASD_COLOR)

    ax.plot(time, km[:, :, 1].mean(0), color=c.EI_COLOR, zorder=3, linewidth=3, label="E/I")
    ax.fill_between(time, km[:, :, 1].mean(0) + km[:, :, 1].std(0), km[:, :, 1].mean(0) - km[:, :, 1].std(0), alpha=0.3,
                    color=c.EI_COLOR)
    ax.hlines(c.LR_THRESHOLD, 0, c.LR_MAX_T, linewidths=5, linestyles=":", colors="black", label="threshold", zorder=5)
    # ax.legend()


def variance_subplot(ax, asd_cv, nt_cv):
    subax: plt.Axes = inset_axes(ax, width="70%", height="60%", loc="lower right")
    subax.plot(nt_cv, color=c.NT_COLOR, zorder=1, label="NDR")
    subax.plot(asd_cv, color=c.ASD_COLOR, zorder=2, label="IDR")
    subax.set_title("Coefficient of variation", fontsize=20)
    subax.set_ylabel(f"Mean CV\nover {c.LR_VAR_WINDOW_SIZE} steps", fontsize=15)
    subax.set_xticklabels(())
    for tick in subax.yaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    subax.legend()
    return subax


def plot_learning_rate(ax, asd_lr, nt_lr):
    min_edge = min(nt_lr.min(), asd_lr.min())
    max_edge = max(nt_lr.max(), asd_lr.max())
    bins = np.linspace(min_edge, max_edge, 20)
    ax.hist(nt_lr, color=c.NT_COLOR, alpha=0.7, label="NDR learning rate", bins=bins)
    ax.hist(asd_lr, color=c.ASD_COLOR, alpha=0.7, label="IDR learning rate", bins=bins)
    ax.legend()


def plot_last_km_distributions(ax, km):
    bins = np.linspace(km[:, -1, :].min(), km[-1, :, :].max(), c.LR_LAST_KM_BIN_NUM)
    ax.hist(km[:, -1, 0], bins=bins, color=c.NT_COLOR, alpha=0.7, label="NDR")
    ax.hist(km[:, -1, 1], bins=bins, color=c.EI_COLOR, alpha=0.7, label="E/I")
    ax.hist(km[:, -1, 2], bins=bins, color=c.ASD_COLOR, alpha=0.7, label="IDR")
    ax.vlines(c.LR_THRESHOLD, 0, km[:, -1, :].max(), colors="k", linestyles=":")


def plot_learning_rate_and_accuracy(km, time, cluster_df,
                                    nt_line_text=(500, 0.45), asd_line_text=(1350, 0.425),
                                    nt_hist_text=(0.4975, 20), asd_hist_text=(0.475, 10),
                                    ei_hist_text=(0.49, 20)) -> [plt.Figure, plt.Axes]:
    lr_fig, axes = plt.subplots(1, 1, figsize=get_fig_size(2, 1.3))
    set_labels(axes, "", "Time (steps)", r"Learned value", title_size=18, ax_label_size=16)
    plot_km_tracking(axes, km, time)
    if cluster_df is not None:
        significant_cluster_df: pd.DataFrame = cluster_df.loc[cluster_df['p-value'] < 0.05,]
        for i, row in significant_cluster_df.iterrows():
            axes.axvspan(row['Start index'], row['End index'], alpha=0.2, color=c.HIST_COLOR)
    subax: plt.Axes = inset_axes(axes, width="100%", height="100%", loc="lower left", bbox_transform=axes.transAxes,
                                 bbox_to_anchor=(0.3, 0.06, 0.65, 0.6))
    plot_last_km_distributions(subax, km)
    subax.set_title("Last learnt value", fontsize=14)
    subax.set_ylabel(f"Frequency", fontsize=14)
    # axes.text(nt_line_text[0], nt_line_text[1], "NDR", color=c.NT_COLOR, size=30, weight="bold")
    # axes.text(asd_line_text[0], asd_line_text[1], "IDR", color=c.ASD_COLOR, size=30, weight="bold")
    subax.text(nt_hist_text[0], nt_hist_text[1], "NDR", color=c.NT_COLOR, size=12, weight="bold")
    subax.text(asd_hist_text[0], asd_hist_text[1], "IDR", color=c.ASD_COLOR, size=12, weight="bold")
    subax.text(ei_hist_text[0], ei_hist_text[1], "E/I", color=c.EI_COLOR, size=12, weight="bold")
    axes.tick_params(axis='both', which='major', labelsize=12, width=2, length=10)
    subax.tick_params(axis='both', which='major', labelsize=10, width=1, length=8)

    return lr_fig, subax


def plot_sensitivity_to_signal_differences(s, nt_resp, asd_resp, ei_resp=None, si=False):
    ei = ei_resp is not None
    fig = plt.figure(figsize=get_fig_size(1, 1.3))
    ax: plt.Axes = fig.subplots()
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.1)
    ax.plot(s, nt_resp, color=c.NT_COLOR, label=f"Sharp, n={c.SI_SSD_NT_N if si else c.SSD_NT_N}", zorder=1,
            linewidth=3)
    ax.plot(s, asd_resp, color=c.ASD_COLOR, label=f"Gradual, n={c.SI_SSD_ASD_N if si else c.SSD_ASD_N}", zorder=2,
            linewidth=3)
    if ei:
        ax.plot(s, ei_resp, color=c.EI_COLOR, label=f"E/I, n={c.SI_SSD_NT_N if si else c.SSD_NT_N}, $\\nu={c.EI_NU}$",
                zorder=3, linewidth=3)
    ax.legend(fontsize=22)
    get_diff = lambda resp: (np.abs(resp[c.SSD_LOW_SIG2_IDX] - resp[c.SSD_LOW_SIG1_IDX]), np.abs(
        resp[c.SSD_HIGH_SIG2_IDX] - resp[c.SSD_HIGH_SIG1_IDX]))
    nt_low_diff, nt_high_diff = get_diff(nt_resp)
    asd_low_diff, asd_high_diff = get_diff(asd_resp)
    if ei:
        ei_low_diff, ei_high_diff = get_diff(ei_resp)

    sig_idx_array = np.array([c.SSD_LOW_SIG1_IDX, c.SSD_LOW_SIG2_IDX, c.SSD_HIGH_SIG1_IDX, c.SSD_HIGH_SIG2_IDX])
    ax.vlines(s[sig_idx_array[:2]], -.1, asd_resp[sig_idx_array[:2]], linestyles=":", colors="black", zorder=3)
    resp = ei_resp if ei else nt_resp
    ax.vlines(s[sig_idx_array[2:]], -.1, resp[sig_idx_array[2:]], linestyles=":", colors="black", zorder=4)
    set_ax_labels(ax, "", "Signal level", r"Neural gain")
    nt_high_diff_str = str(utils.num2print(nt_high_diff))
    asd_high_diff_str = str(utils.num2print(asd_high_diff))
    nt_low_diff_str = str(utils.num2print(nt_low_diff))
    asd_low_diff_str = str(utils.num2print(asd_low_diff))
    if ei:
        text = "$\\Delta A_{EI}=%s$\n$\\Delta A_{NDR}=%s$\n$\\Delta A_{IDR}=%s$"
        ei_high_diff_str = str(utils.num2print(ei_high_diff))
        ei_low_diff_str = str(utils.num2print(ei_low_diff))
        cd_vals = (ei_high_diff_str.replace("-", "^{-") + ("}" if "-" in ei_high_diff_str else ""),
                   nt_high_diff_str.replace("-", "^{-") + ("}" if "-" in nt_high_diff_str else ""),
                   asd_high_diff_str.replace("-", "^{-") + ("}" if "-" in asd_high_diff_str else ""))
        ab_vals = (ei_low_diff_str.replace("-", "^{-") + ("}" if "-" in ei_low_diff_str else ""),
                   nt_low_diff_str.replace("-", "^{-") + ("}" if "-" in nt_low_diff_str else ""),
                   asd_low_diff_str.replace("-", "^{-") + ("}" if "-" in asd_low_diff_str else ""))

    else:
        text = "$\\Delta A_{NDR}=%s$\n$\\Delta A_{IDR}=%s$"
        cd_vals = (nt_high_diff_str.replace("-", "^{-") + ("}" if "-" in nt_high_diff_str else ""),
                   asd_high_diff_str.replace("-", "^{-") + ("}" if "-" in asd_high_diff_str else ""))
        ab_vals = (nt_low_diff_str.replace("-", "^{-") + ("}" if "-" in nt_low_diff_str else ""),
                   asd_low_diff_str.replace("-", "^{-") + ("}" if "-" in asd_low_diff_str else ""))

    ax.text(0.81, 0.65, text % cd_vals, fontsize=20)
    ax.text(0, 0.1, text % ab_vals, fontsize=20)
    ax.text(0.59, 0.69, "IDR", color=c.ASD_COLOR, fontsize=24, weight='bold')
    if ei:
        ax.text(0.46, 0.9, "E/I", color=c.EI_COLOR, fontsize=24, weight='bold')
    if ei:
        ax.text(0.62, 1.03, "NDR", color=c.NT_COLOR, fontsize=24, weight='bold')
    else:
        ax.text(0.46, 0.9, "NDR", color=c.NT_COLOR, fontsize=24, weight='bold')
    ax.text(0.2, -0.175, "A", fontsize=20, fontweight='bold', horizontalalignment='center')
    ax.text(0.3, -0.175, "B", fontsize=20, fontweight='bold', horizontalalignment='center')
    ax.text(0.7, -0.175, "C", fontsize=20, fontweight='bold', horizontalalignment='center')
    ax.text(0.8, -0.175, "D", fontsize=20, fontweight='bold', horizontalalignment='center')
    return fig


def plot_sensitivity_to_signal_differences_rosenberg(s, nt_resp, asd_resp, ax=None):
    if ax is None:
        fig = plt.figure(figsize=get_fig_size(1, 1.5))
        ax: plt.Axes = fig.subplots()
    else:
        fig = ax.get_figure()
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.1)
    ax.plot(s, nt_resp, color=c.NT_COLOR, label=f"Full inhibition, c=1", zorder=1, linewidth=3)
    ax.plot(s, asd_resp, color=c.EI_COLOR, label=f"Decreased inhibition, c=0.75", zorder=2, linewidth=3)
    ax.legend(fontsize=22)
    get_diff = lambda resp: (np.abs(resp[c.SSD_LOW_SIG2_IDX] - resp[c.SSD_LOW_SIG1_IDX]), np.abs(
        resp[c.SSD_HIGH_SIG2_IDX] - resp[c.SSD_HIGH_SIG1_IDX]))
    nt_low_diff, nt_high_diff = get_diff(nt_resp)
    asd_low_diff, asd_high_diff = get_diff(asd_resp)

    sig_idx_array = np.array([c.SSD_LOW_SIG1_IDX, c.SSD_LOW_SIG2_IDX, c.SSD_HIGH_SIG1_IDX, c.SSD_HIGH_SIG2_IDX])
    ax.vlines(s[sig_idx_array[:2]], -.1, asd_resp[sig_idx_array[:2]], linestyles=":", colors="black", zorder=3)
    ax.vlines(s[sig_idx_array[2:]], -.1, asd_resp[sig_idx_array[2:]], linestyles=":", colors="black", zorder=4)
    set_ax_labels(ax, "", "Signal level", r"Neural gain")
    text = "$\\Delta A_{Full}=%s$\n$\\Delta A_{Decreased}=%s$"
    cd_vals = (str(utils.num2print(nt_high_diff)).replace("-", "^{-") + "}",
               str(utils.num2print(asd_high_diff)).replace("-", "^{-") + "}")
    ab_vals = (str(utils.num2print(nt_low_diff)).replace("-", "^{-") + "}",
               str(utils.num2print(asd_low_diff)).replace("-", "^{-") + "}")

    ax.text(81, 0.45, text % cd_vals, fontsize=20)
    ax.text(30, 0.1, text % ab_vals, fontsize=20)
    ax.text(20.1, 1.21, "Decreased inhibition", color=c.EI_COLOR, fontsize=18, weight='bold')
    ax.text(41, 0.78, "Full inhibition", color=c.NT_COLOR, fontsize=18, weight='bold')
    ax.text(20, -0.175, "A", fontsize=20, fontweight='bold', horizontalalignment='center')
    ax.text(30, -0.175, "B", fontsize=20, fontweight='bold', horizontalalignment='center')
    ax.text(70, -0.175, "C", fontsize=20, fontweight='bold', horizontalalignment='center')
    ax.text(80, -0.175, "D", fontsize=20, fontweight='bold', horizontalalignment='center')
    return fig


def plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate, ei_sig_estimate=None, text_x=(160, 25),
                       text_y=(90, 300), fig=None, axs=None, labels=None) -> plt.Figure:
    ei = ei_sig_estimate is not None
    if labels is None:
        labels = ["IDR", "NDR", "E/I"]
    if fig is None:
        fig = plt.figure(figsize=get_fig_size(1, 1.2))
        axs = fig.subplots()
    # plt.subplots_adjust(left=c.LEFT_SPACE + 0.05)
    change_timepoint = round(max(c.SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))

    asd_times = np.argmax(np.abs(asd_sig_estimate) >= c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX, axis=1) - change_timepoint
    nt_times = np.argmax(np.abs(nt_sig_estimate) >= c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX, axis=1) - change_timepoint
    if ei:
        ei_times = np.argmax(np.abs(ei_sig_estimate) >= c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX, axis=1) - change_timepoint
    mins = [asd_times.min(), nt_times.min()]
    maxs = [asd_times.max(), nt_times.max()]
    if ei:
        mins += [ei_times.min()]
        maxs += [ei_times.max()]
    min_bin = min(mins)
    max_bin = max(maxs)
    bins = np.linspace(min_bin, max_bin, 50)
    l1, = axs.plot(asd_sig_estimate[c.SR_PLOT_REP_IDX, :], label=labels[0], color=c.ASD_COLOR, linewidth=3)
    l2, = axs.plot(nt_sig_estimate[c.SR_PLOT_REP_IDX, :], label=labels[1], color=c.NT_COLOR, linewidth=2, alpha=0.7)
    if ei:
        l3, = axs.plot(ei_sig_estimate[c.SR_PLOT_REP_IDX, :], label=labels[2], color=c.EI_COLOR, linewidth=2, alpha=0.7)

    l4, = axs.plot(signal[c.SR_PLOT_REP_IDX, :, 0], color='gray', label="measured signal", alpha=0.3)
    l5, = axs.plot([c.SR_SIG_MIN] * change_timepoint +
                   [c.SR_SIG_MAX] * (c.SR_NUM_STEPS - change_timepoint), color='k',
                   label="real signal", linewidth=3, zorder=1)

    subax = inset_axes(axs, "40%", "40%", loc="lower right", borderpad=3,
                       bbox_transform=axs.transAxes)  # type: plt.Axes
    subax.grid(False)
    subax.hist(asd_times, bins=bins, label=labels[0], color=c.ASD_COLOR, alpha=0.6)
    subax.hist(nt_times, bins=bins, label=labels[1], color=c.NT_COLOR, alpha=0.6)
    if ei:
        subax.hist(ei_times, bins=bins, label=labels[2], color=c.EI_COLOR, alpha=0.6)
    subax.text(text_x[0], text_y[0], labels[0], color=c.ASD_COLOR, size=20, weight="bold")
    subax.text(text_x[1], text_y[1], labels[1], color=c.NT_COLOR, size=20, weight="bold")
    if ei:
        subax.text(text_x[1], text_y[1] - 100, labels[2], color=c.EI_COLOR, size=20, weight="bold")
    axs.set_xlabel("Time (steps)")
    axs.set_ylabel("Signal")
    subax.set_xlabel("Time after switch (steps)")
    subax.set_ylabel("Frequency")
    if ei:
        subax.legend([l1, l2, l3, l4, l5], labels + ["Noisy signal", "Signal"], loc="upper right",
                     bbox_to_anchor=(1.1, 1.6), fontsize=13)
    else:
        subax.legend([l1, l2, l4, l5], labels[:2] + ["Noisy Signal", "Signal"], loc="upper right",
                     bbox_to_anchor=(1.1, 1.6), fontsize=13)

    return fig


def plot_binocular_rivalry(asd_transition_count, nt_transition_count, ei_transition_count, asd_ratios, nt_ratios,
                           ei_ratios, asd_resp, nt_resp, ei_resp, signal, example_idx=0) -> plt.Figure:
    fig: plt.Figure = plt.figure(figsize=get_fig_size(1.5, 2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.65, 0.35])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    # fig, axes = get_3_axes_with_1st_full()  # fig.subplots(2, 1)
    fig.subplots_adjust(hspace=c.HSPACE + 1, wspace=c.WSPACE + 0.2, left=c.LEFT_SPACE + 0.01,
                        bottom=c.BOTTOM_SPACE - 0.02)
    max_transitions = max(np.max(nt_transition_count), np.max(asd_transition_count), np.max(ei_transition_count))
    max_ratios = 50
    axes[0].grid(False)
    axes[0].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    axes[1].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    axes[2].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)

    axes[0].tick_params(axis='both', which='minor', width=3, length=12)
    axes[1].tick_params(axis='both', which='minor', width=3, length=12)
    axes[2].tick_params(axis='both', which='minor', width=3, length=12)

    axes[1].hist(ei_transition_count, color=c.EI_COLOR, bins=np.arange(0, max_transitions, 2), label="E/I", alpha=0.7)
    axes[1].hist(asd_transition_count, color=c.ASD_COLOR, bins=np.arange(0, max_transitions, 2), label="IDR", alpha=0.7)
    axes[1].hist(nt_transition_count, color=c.NT_COLOR, bins=np.arange(0, max_transitions, 2), label="NDR", alpha=0.7)
    axes[1].text(5, 150, "IDR", color=c.ASD_COLOR, fontsize=35, weight='bold')
    axes[1].text(8, 100, "NDR", color=c.NT_COLOR, fontsize=35, weight='bold')
    axes[1].text(18, 60, "E/I", color=c.EI_COLOR, fontsize=35, weight='bold')
    # axes[1].legend(fontsize=30)

    axes[2].hist(ei_ratios, label="E/I", bins=np.arange(0, max_ratios, .5), alpha=0.7, color=c.EI_COLOR)
    axes[2].hist(asd_ratios, label="IDR", bins=np.arange(0, max_ratios, .5), alpha=0.7, color=c.ASD_COLOR)
    axes[2].hist(nt_ratios, label="NDR", bins=np.arange(0, max_ratios, .5), alpha=0.7, color=c.NT_COLOR)
    # axes[2].legend(fontsize=30)
    axes[2].text(4, 70, "IDR", color=c.ASD_COLOR, fontsize=35, weight='bold')
    axes[2].text(8, 30, "NDR", color=c.NT_COLOR, fontsize=35, weight='bold')
    axes[2].text(20, 20, "E/I", color=c.EI_COLOR, fontsize=35, weight='bold')

    axes[0].plot(signal[example_idx, :, 0], linewidth=4, color='k', label="Signal")
    axes[0].plot(ei_resp[example_idx, :], alpha=0.5, linewidth=4, color=c.EI_COLOR, label="E/I")
    axes[0].plot(asd_resp[example_idx, :], alpha=0.5, linewidth=4, color=c.ASD_COLOR, label="IDR")
    axes[0].plot(nt_resp[example_idx, :], alpha=0.5, linewidth=4, color=c.NT_COLOR, label="NDR")
    axes[0].hlines([0.8, 0.2], xmin=0, xmax=nt_resp[example_idx, :].size, alpha=0.5, linewidth=5, linestyle=":",
                   color='k')
    axes[0].legend(fontsize=30)
    set_labels(axes, "", ["Time (steps)", "# of transitions", r"pure state ratio $\frac{t_{pure}}{t_{mixed}}$"],
               ["Population response", "Frequency", ""])

    return fig


def plot_binocular_rivalry_rosenberg(asd_transition_count, nt_transition_count, asd_ratios, nt_ratios,
                                     asd_resp, nt_resp, signal, example_idx=0) -> plt.Figure:
    fig: plt.Figure = plt.figure(figsize=get_fig_size(3, 4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.65, 0.35])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    max_transitions = max(np.max(nt_transition_count), np.max(asd_transition_count))
    max_ratios = min(50, max(np.max(asd_ratios), np.max(nt_ratios)))
    axes[0].grid(False)
    axes[0].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    axes[1].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    axes[2].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)

    axes[0].tick_params(axis='both', which='minor', width=3, length=12)
    axes[1].tick_params(axis='both', which='minor', width=3, length=12)
    axes[2].tick_params(axis='both', which='minor', width=3, length=12)

    axes[1].hist(asd_transition_count, color=c.ASD_COLOR, bins=np.arange(0, max_transitions + 2, 1),
                 label="Decreased inhibition",
                 alpha=0.7)
    axes[1].hist(nt_transition_count, color=c.NT_COLOR, bins=np.arange(0, max_transitions + 2, 1),
                 label="Full inhibition",
                 alpha=0.7)
    axes[1].text(4.2, 75, "Decreased inhibition", color=c.ASD_COLOR, fontsize=35, weight='bold')
    axes[1].text(2.1, 160, "Full inhibition", color=c.NT_COLOR, fontsize=35, weight='bold')
    # axes[1].legend(fontsize=30)

    axes[2].hist(asd_ratios, label="Decreased inhibition", bins=np.arange(0, max_ratios + 1, .5), alpha=0.7,
                 color=c.ASD_COLOR)
    axes[2].hist(nt_ratios, label="Full inhibition", bins=np.arange(0, max_ratios + 1, .5), alpha=0.7,
                 color=c.NT_COLOR)
    # axes[2].legend(fontsize=30)
    axes[2].text(2.5, 50, "Decreased inhibition", color=c.ASD_COLOR, fontsize=35, weight='bold')
    axes[2].text(1, 200, "Full inhibition", color=c.NT_COLOR, fontsize=35, weight='bold')

    axes[0].plot(signal[example_idx, :, 0], linewidth=4, color='k', label="Signal")
    axes[0].plot(asd_resp[example_idx, :], alpha=0.5, linewidth=4, color=c.ASD_COLOR, label="Decreased inhibition")
    axes[0].plot(nt_resp[example_idx, :], alpha=0.5, linewidth=4, color=c.NT_COLOR, label="Full inhibition")
    axes[0].hlines([0.6, 0.2], xmin=0, xmax=nt_resp[example_idx, :].size, alpha=0.5, linewidth=5, linestyle=":",
                   color='k')
    axes[0].legend(fontsize=30)
    set_labels(axes, "", ["Time (steps)", "# of transitions", r"pure state ratio $\frac{t_{pure}}{t_{mixed}}$"],
               ["Population response", "Frequency", ""])

    return fig


def heatmap(data_matrix, xvals, yvals, text=True, text_size=15, colorbar=True, cmap=plt.cm.viridis, precision=2):
    fig = plt.figure(figsize=get_fig_size(1.5, 1.5))
    ax: plt.Axes = fig.subplots()
    cbar = ax.imshow(data_matrix, cmap=cmap, aspect='auto')
    plt.xticks(np.arange(xvals.size), np.round(xvals, precision))
    plt.yticks(np.arange(yvals.size), np.round(yvals, precision)[::-1])
    if text:
        for i in range(yvals.size):
            for j in range(xvals.size):
                ax.text(j, i, f"{np.round(data_matrix[i, j], precision)}",
                        ha="center", va="center", color="k", fontsize=text_size)
    if colorbar:
        cbar = plt.colorbar(cbar)
        cbar.ax.tick_params(labelsize=20)
    return fig, ax


def save_all_separatrix_plots(time, signal, signal_sigma_levels, km_sigma_levels, activated_frac_mat):
    for noise_level_idx in range(signal_sigma_levels.size):
        fig = plt.figure(figsize=get_fig_size(1.5, 1.8))
        ax: plt.Axes = fig.subplots()
        ax.set_xlabel('Heterogeneity')
        ax.set_ylabel('signal level')
        ax.set_title('As heterogeneity increases\nthe probability of activation in\nlow signal levels increases')

        cbar = ax.imshow(activated_frac_mat[:, noise_level_idx, :, -1], cmap=plt.cm.Reds, aspect='auto')
        plt.xticks(np.arange(km_sigma_levels.size), np.round(km_sigma_levels, 2))
        plt.yticks(np.arange(signal.size), np.round(signal, 2)[::-1])
        for i in range(signal.size):
            for j in range(km_sigma_levels.size):
                ax.text(j, i, f"{activated_frac_mat[i, noise_level_idx, j, -1]:.2f}",
                        ha="center", va="center", color="k", fontsize=15)
        ax.text(-.25, -1.2, f"time={time[-1]}", ha="center", va="center", color="k", fontsize=23,
                fontweight='bold')
        cbar = plt.colorbar(cbar)
        cbar.ax.tick_params(labelsize=20)
        savefig(fig, f"Separatrix, n_neurons={c.SEP_N_NEURONS}, noise={signal_sigma_levels[noise_level_idx]:.2f}",
                shift_y=1.1, shift_x=-.11, si=True, tight=False)
        plt.close(fig)


def plot_separatrix(signal_noise, activated_frac_mat, effective_n, ci, ax=None):
    fig = None
    color_list = get_color_mix(c.NT_COLOR, c.ASD_COLOR, len(effective_n))
    if ax is None:
        fig, ax = plt.subplots(figsize=get_fig_size(1, 1.5))  # pl.get_3_axes_with_3rd_centered()
    for i, n in enumerate(effective_n):
        ax.plot(signal_noise, 100 * activated_frac_mat[:, i], color=color_list[i],
                label="$n_{eff}=%d$" % round(n), linewidth=5)
        ax.fill_between(signal_noise, 100 * ci.low[:, i], 100 * ci.high[:, i], color=color_list[i], alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    set_ax_labels(ax, "", "Signal noise", "Activation probability")
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.1)
    ax.legend(fontsize=30)
    if fig is not None:
        return fig, ax


def plot_n_fi(fi, s, si=False, ax: plt.Axes = None):
    fi_n_list = c.SI_FI_N_LIST if si else c.FI_N_LIST
    s_dt = s[1] - s[0]
    mix_colors = get_normed_color_mix(c.ASD_COLOR, c.NT_COLOR, fi_n_list)
    if ax is None:
        fig = plt.figure()
        fig.subplots_adjust(left=c.LEFT_SPACE + 0.08)
        ax = fig.subplots()
    subax: plt.Axes = inset_axes(ax, width="40%", height="60%")
    subax.set_xlabel("Hill-coefficient\nn", fontsize=23)
    subax.set_ylabel(r"$\int_0^1 I_{F}(S) dS$", fontsize=23)
    total_enc_capacity = []
    for i, enc_cap in enumerate(fi):
        ax.plot(s, np.sqrt(enc_cap), color=mix_colors[i], linewidth=4,
                label="$n=%d$" % (fi_n_list[i]))
        total_enc_capacity.append(enc_cap.sum() * s_dt)
    subax.bar(fi_n_list, total_enc_capacity, width=0.8 * (fi_n_list[1] - fi_n_list[0]), color=mix_colors)
    subax.set_xticks(fi_n_list)
    subax.tick_params(axis='both', which='major', labelsize=20)
    subax.minorticks_off()
    subax.grid(False)
    set_ax_labels(ax, "Dynamic range", "Signal level", r"Encoding capacity $\sqrt{I_{F}(S)}$", title_size=30,
                  ax_label_size=30)
    ax.legend(fontsize=25, loc="upper left")
    ax.tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    ax.set_xlim(-0.1, 1.35)
    ax.set_xticks(np.linspace(0, 1, 6))
    subax.tick_params(axis='both', which='major', width=3, length=15)
    return subax


def plot_ei_fi(fi, s, ax=None):
    fi_nu_list = c.FI_NU_LIST
    s_dt = s[1] - s[0]
    mix_colors = get_normed_color_mix(c.ASD_COLOR, c.NT_COLOR, fi_nu_list)
    if ax is None:
        fig = plt.figure()
        fig.subplots_adjust(left=c.LEFT_SPACE + 0.08)
        ax = fig.subplots()
    subax: plt.Axes = inset_axes(ax, width="40%", height="60%")
    subax.set_xlabel("E/I factor\n$\\nu$", fontsize=23)
    subax.set_ylabel(r"$\int_0^1I_F(S)dS$", fontsize=23)
    total_enc_capacity = []
    for i, enc_cap in enumerate(fi):
        ax.plot(s, np.sqrt(enc_cap), color=mix_colors[i], linewidth=4,
                label="$\\nu=%.1f$" % (fi_nu_list[i]))
        total_enc_capacity.append(enc_cap.sum() * s_dt)
    subax.bar(fi_nu_list, total_enc_capacity, width=0.8 * (fi_nu_list[1] - fi_nu_list[0]), color=mix_colors)
    subax.set_xticks(fi_nu_list)
    subax.tick_params(axis='both', which='major', labelsize=20)
    subax.minorticks_off()
    subax.grid(False)
    set_ax_labels(ax, "Inhibition strength", "Signal level", "", title_size=30,
                  ax_label_size=30)
    ax.legend(fontsize=25, loc="upper left")
    ax.tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    ax.set_xlim(-0.1, 1.35)
    ax.set_xticks(np.linspace(0, 1, 6))
    subax.tick_params(axis='both', which='major', width=3, length=15)
    return subax


def plot_learning_as_function_of_n(n_range, n_range_km, time):
    fig, axes = get_3_axes_with_3rd_centered()
    axes[0].set(title="Learning rate increases with slope (n)", xlabel="n", ylabel=r"$\frac{1}{\tau}$")
    axes[0].set_ylabel(r"$\frac{1}{\tau}$", fontsize=50)
    axes[1].set(title="Bias decreases with slope (n)", xlabel="n", ylabel=r"$\theta - \hat{\theta}$")
    axes[2].set(title="Bias standard deviation decreases with slope (n)", xlabel="n",
                ylabel=r"$\sigma (\theta - \hat{\theta})$")
    n_range_lr = np.argmax(n_range_km >= c.LR_THRESHOLD * c.LR_THRESHOLD_PERCENTAGE, axis=1)
    bias = np.abs(n_range_km[:, -1, :] - c.LR_THRESHOLD)
    axes[0].plot(n_range, n_range_lr.mean(0), '-o')
    axes[1].plot(n_range, bias.mean(0), '-o')
    axes[2].plot(n_range, bias.std(0), '-o')
    make_log_scale(axes)
    return fig


def plot_time_to_track_as_a_function_of_n(time_to_track, n_range):
    fig, ax = plt.subplots(figsize=get_fig_size(1., 1.3))
    ax: plt.Axes
    ax.set(title="Time to 95% of signal after change as a function of slope",
           xlabel="n", ylabel="Time to 95% of signal after change")
    ax.errorbar(n_range.mean(-1), time_to_track.mean(-1), xerr=time_to_track.std(-1), marker='o', markersize=5)
    return fig


def plot_dynamic_range_as_function_of_slope(slopes, widths, ratios):
    fig, axes = plt.subplots(1, 2, figsize=get_fig_size(0.7, 2.))
    axes[0].set(title="Response width VS slope (n)", xlabel="n", ylabel=r"$S_{0.9}-S_{0.1}$")
    axes[0].plot(slopes, widths[0])
    axes[0].fill_between(slopes, widths[1], widths[2], alpha=0.3)
    axes[1].set(title="Dynamic range ratio (R) VS slope (n)", xlabel="n", ylabel=r"$\frac{S_{0.9}}{S_{0.1}}$")
    axes[1].plot(slopes, ratios[0])
    axes[1].fill_between(slopes, ratios[1], ratios[2], alpha=0.3)
    # make_log_scale(axes)
    return fig


def plot_slower_updating(signal, asd_sig_estimate, nt_sig_estimate, asd_data, nt_data):
    fig: plt.Figure = plt.figure(figsize=(17, 10))
    gs = GridSpec(7, 9, fig, top=0.93, bottom=0.1, hspace=0.2, wspace=0.2, height_ratios=[1, 1, 1, 1, 1, 1, 1],
                  width_ratios=[1, 1, 1, 1, 0.5, 1, 1, 1, 1])
    kalman_ax = fig.add_subplot(gs[0:3, 0:4])
    hist_ax = fig.add_subplot(gs[0:3, 5:])
    asd_n = asd_data.get_median_hill_coefficients()
    asd_n = asd_n[~np.isnan(asd_n)]
    nt_n = nt_data.get_median_hill_coefficients()
    nt_n = nt_n[~np.isnan(nt_n)]
    plot_hill_coefficient_distribution(asd_n, nt_n, hist_ax)
    asd_axes = []
    nt_axes = []
    for i in range(3):
        asd_axes.append([])
        nt_axes.append([])
        for j in range(2):
            asd_axes[i].append(fig.add_subplot(gs[4 + i, 2 * j:(2 * j) + 2]))
            nt_axes[i].append(fig.add_subplot(gs[4 + i, 5 + 2 * j:(2 * j) + 7]))
            nt_axes[i][j].get_yaxis().set_ticklabels([])
            if j != 0:
                asd_axes[i][j].get_yaxis().set_ticklabels([])
            if i != 2:
                nt_axes[i][j].get_xaxis().set_ticklabels([])
                asd_axes[i][j].get_xaxis().set_ticklabels([])
    plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate, fig=fig, axs=kalman_ax)
    plot_full_data_response_vs_fit(asd_data, c_model=c.HIST_COLOR, fig=fig, axes=np.array(asd_axes))
    plot_full_data_response_vs_fit(nt_data, c_model=c.HIST_COLOR, fig=fig, axes=np.array(nt_axes))

    return fig


def plot_tapping_tracking_delays(asd_tau, nt_tau):
    fig, ax = plt.subplots()
    violin_res = ax.violinplot([nt_tau, asd_tau], showmeans=False, showmedians=True, widths=0.7)
    colors = [c.DARK_NT_COLOR, c.DARK_ASD_COLOR]
    for pc, color in zip(violin_res['bodies'], colors):
        pc.set_facecolor(color)
    violin_res['cmedians'].set_colors(colors)
    violin_res['cbars'].set_colors(colors)
    violin_res['cmins'].set_colors(colors)
    violin_res['cmaxes'].set_colors(colors)
    np.random.seed(97)
    ax.scatter(np.random.normal(1, 0.05, nt_tau.shape), nt_tau, color=colors[0], facecolor='none')
    ax.scatter(np.random.normal(2, 0.05, asd_tau.shape), asd_tau, color=colors[1], facecolor='none')
    ax.set_xticks([1, 2], labels=["NT", "ASD"])
    ax.set(title="Dynamics tracking lag", ylabel="Lag", xlabel="Group")
    return fig


def plot_hill_coefficient_distribution(asd_n, nt_n, ax):
    ax.hist(asd_n, color=c.DARK_ASD_COLOR, alpha=0.5, label="ASD", density=True)
    ax.hist(nt_n, color=c.DARK_NT_COLOR, alpha=0.5, label="NT", density=True)
    ax.set_xlabel("Hill Coefficients")
    ax.set_ylabel("Density")
    t, p = scipy.stats.mannwhitneyu(asd_n, nt_n, alternative="less")
    ax: plt.Axes = plt.gca()
    # ax.text(0.2, 0.9, r"$n_{ASD} < n_{NT}$", transform=ax.transAxes, fontsize=25)
    # ax.text(0.2, 0.8, "U(%d)=%d, p<%.1g" % (asd_n.size + nt_n.size - 2, t, p), transform=ax.transAxes, fontsize=20)
    ax.legend()


def plot_total_fi_fit(asd_fi, nt_fi, angeliki_nt_total_fi, angeliki_asd_total_fi, ax):
    ax.hist(nt_fi, color=c.NT_COLOR, label="NT fit", alpha=0.6, density=True)
    ax.hist(angeliki_nt_total_fi, color=c.DARK_NT_COLOR, label="NT data", alpha=0.6, density=True)

    ax.hist(asd_fi, color=c.ASD_COLOR, label="ASD fit", alpha=0.6, density=True)
    ax.hist(angeliki_asd_total_fi, color=c.DARK_ASD_COLOR, label="ASD data", alpha=0.6, density=True)

    set_ax_labels(ax, "Total FI Distribution", r"Total $I_F$", "", title_size=30,
                  ax_label_size=30)
    ax.legend(fontsize=25, loc="upper left")
    ax.tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    ax.legend(fontsize=20)


def plot_total_fi_hill_coeffs_fit(asd_n, nt_n, ax):
    ax.hist(nt_n, color=c.NT_COLOR, label="NDR", alpha=0.6, density=True)
    ax.hist(asd_n, color=c.ASD_COLOR, label="IDR", alpha=0.6, density=True)
    set_ax_labels(ax, "Hill Coefficients Distribution", r"Hill coefficient", "Density", title_size=30,
                  ax_label_size=30)
    ax.legend(fontsize=25, loc="upper left")
    ax.tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    ax.legend(fontsize=20)


def plot_fi(fi, ei_fi, nt_n, asd_n, nt_total_fi, asd_total_fi, angeliki_nt_total_fi, angeliki_asd_total_fi, s):
    fig, axes = plt.subplots(2, 2, figsize=get_fig_size(2, 3), sharey='row')

    subax1 = plot_n_fi(fi, s, ax=axes[0, 0])
    subax2 = plot_ei_fi(ei_fi, s, ax=axes[0, 1])

    plot_total_fi_hill_coeffs_fit(asd_n, nt_n, axes[1, 0])
    plot_total_fi_fit(asd_total_fi, nt_total_fi, angeliki_nt_total_fi, angeliki_asd_total_fi, axes[1, 1])

    return fig, subax1, subax2


def plot_si_fi(fi, ei_fi, s):
    fig, axes = plt.subplots(1, 2, figsize=get_fig_size(1.5, 3), sharey='row')

    subax1 = plot_n_fi(fi, s, ax=axes[0], si=True)
    subax2 = plot_ei_fi(ei_fi, s, ax=axes[1])
    return fig, subax1, subax2


def combine_plots(filenames, output_filename, wspace=0.05, hspace=0.05):
    n_rows = int(np.floor(np.sqrt(len(filenames))))
    n_cols = int(np.ceil(np.sqrt(len(filenames))))
    centered_last_row = n_cols * n_rows < len(filenames)

    images = list(
        map(lambda x: Image.open if x.endswith('png') or x.endswith('jpg') else convert_from_path(x)[0], filenames))
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    # get font size

    total_width = int(np.ceil(n_cols * (max_width + wspace * max_width)))
    total_height = int(np.ceil((n_rows + (1 * centered_last_row)) * (max_height + hspace * max_height)))
    new_im = Image.new('RGB', (total_width, total_height), color="#FFFFFF")
    draw = ImageDraw.Draw(new_im)
    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            im = images[i]

            x_offset = int(np.ceil(col * max_width + ((col + 1) * max_width * wspace)))
            y_offset = int(np.ceil(row * max_height + ((row + 1) * max_height * hspace)))

            new_im.paste(im, (x_offset, y_offset))

            text = chr(ord('A') + i)
            draw.text((x_offset, y_offset), text, fill="#000000")
            i += 1
    if centered_last_row:
        remaining = images[i:]
        free_space = total_width - max_width * len(remaining) - (len(remaining) + 1) * max_width * wspace
        for j, im in enumerate(remaining):
            col = j % n_cols
            row = n_rows
            x_offset = int(np.ceil(col * max_width + ((col + 1) * max_width * wspace) + free_space / 2))
            y_offset = int(np.ceil(row * max_height + ((row + 1) * max_height * hspace)))

            new_im.paste(im, (x_offset, y_offset))

            text = chr(ord('A') + i)
            draw.text((x_offset, y_offset), text, fill="#000000")
            i += 1
    new_im.save(output_filename)


def plot_tapping_fit(data, color=c.EI_COLOR, title=None):
    fig, axes = plt.subplots(nrows=data.group_dynamics_mean.shape[0] - 2, ncols=data.group_dynamics_mean.shape[1],
                             sharex='all', sharey='row', figsize=(5.5 * data.subjects[0].dynamics.shape[1],
                                                                  2 * data.subjects[0].dynamics.shape[0]))
    change_idx = np.where(data.lags == 0)[0][0]
    true_freq = np.zeros_like(data.lags)
    for i, row_axes in enumerate(axes[::-1]):
        i += 2
        row_axes[0].set_ylabel(f"{int(data.diffs[i])}Hz")
        for j, ax in enumerate(row_axes):
            ax: plt.Axes
            ax.errorbar(data.lags, data.group_dynamics_mean[i, j], data.group_dynamics_std[i, j], None, '-o',
                        ecolor="k", label="real", color="gray")
            ax.plot(data.lags, data.simulated_dynamics_mean[i, j], '-o', color=color, label="model")
            true_freq[:change_idx] = 500 + (((-1) ** j) * (data.diffs[i] / 2))
            true_freq[change_idx:] = 500 + (((-1) ** (j + 1)) * (data.diffs[i] / 2))
            ax.plot(data.lags, true_freq, linestyle=':', color='k')
            ax.set_ylim(true_freq.min() - 10, true_freq.max() + 10)
            xtext, ytext = (0.95, 0.90) if j == 0 else (0.95, 0.1)
            try:
                text = r"$n={n:.2g}, \nu = {nu:.2g}$".format(n=data.fitted_n[i, j], nu=data.fitted_nu[i, j])
            except Exception:
                text = r"$n={n:.2g}$".format(n=data.fitted_n[i, j])
            ax.text(xtext, ytext, text,
                    ha='right', va='center', transform=ax.transAxes, fontsize=20, fontweight='bold')
    axes[0, 0].set_title("Accelerating")
    axes[0, 1].set_title("Decelerating")
    axes[-1, 0].set_xlabel("Lags")
    axes[-1, 1].set_xlabel("Lags")
    axes[0, 0].legend(bbox_to_anchor=[0.65, 0.5], loc="lower left")
    axes[-1, 0].set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    axes[-1, 1].set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    if title:
        fig.suptitle(title)
    return fig


def plot_tapping_fit_rosenberg(data, color=c.EI_COLOR, title=None):
    fig, axes = plt.subplots(nrows=data.group_dynamics_mean.shape[0] - 2, ncols=data.group_dynamics_mean.shape[1],
                             sharex='all', sharey='row', figsize=(5.5 * data.subjects[0].dynamics.shape[1],
                                                                  2 * data.subjects[0].dynamics.shape[0]))
    change_idx = np.where(data.lags == 0)[0][0]
    true_freq = np.zeros_like(data.lags)
    for i, row_axes in enumerate(axes[::-1]):
        i += 2
        row_axes[0].set_ylabel(f"{int(data.diffs[i])}Hz")
        for j, ax in enumerate(row_axes):
            ax: plt.Axes
            ax.errorbar(data.lags, data.group_dynamics_mean[i, j], data.group_dynamics_std[i, j], None, '-o',
                        ecolor="k", label="real", color="gray")
            ax.plot(data.lags, data.simulated_dynamics_mean[i, j], '-o', color=color, label="model")
            true_freq[:change_idx] = 500 + (((-1) ** j) * (data.diffs[i] / 2))
            true_freq[change_idx:] = 500 + (((-1) ** (j + 1)) * (data.diffs[i] / 2))
            ax.plot(data.lags, true_freq, linestyle=':', color='k')
            ax.set_ylim(true_freq.min() - 10, true_freq.max() + 10)
            xtext, ytext = (0.95, 0.90) if j == 0 else (0.95, 0.1)
            text = r"$\nu = {nu:.2g}$".format(n=data.rosenberg_fitted_nu[i, j], nu=data.rosenberg_fitted_nu[i, j])
            ax.text(xtext, ytext, text, ha='right', va='center', transform=ax.transAxes, fontsize=20, fontweight='bold')
    axes[0, 0].set_title("Accelerating")
    axes[0, 1].set_title("Decelerating")
    axes[-1, 0].set_xlabel("Lags")
    axes[-1, 1].set_xlabel("Lags")
    axes[0, 0].legend(bbox_to_anchor=[0.65, 0.5], loc="lower left")
    axes[-1, 0].set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    axes[-1, 1].set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    if title:
        fig.suptitle(title)
    return fig


def plot_variance_width_over_dynamic_range(n_space, widths, stds):
    fig, ax = plt.subplots(figsize=get_fig_size(1.2, 1.5))
    shaded_errplot(n_space, widths, stds, ax=ax, plot_kwargs=dict(marker='o'),
                   labels={"xlabel": "Population Hill coefficient", "ylabel": "Width of variance curve",
                           "title": "Effect of dynamic range on variance curve"})
    ax.set_xlabel("Population Hill coefficient")
    ax.set_ylabel("Width of variance curve")
    ax.set_title("Effect of dynamic range on variance curve")
    return fig


def shaded_errplot(x, y, yerr, ax=None, plot_kwargs=None, shade_kwargs=None, labels=None):
    if shade_kwargs is None:
        shade_kwargs = dict()
    if plot_kwargs is None:
        plot_kwargs = dict()
    if "color" in plot_kwargs and "color" not in shade_kwargs:
        shade_kwargs["color"] = plot_kwargs["color"]
    if ax is None:
        fig, ax = plt.subplots(figsize=get_fig_size(1, 1.2))
    ax.plot(x, y, **plot_kwargs)
    try:
        iter(yerr)
        ax.fill_between(x, yerr[0], yerr[1], alpha=0.5, **shade_kwargs)
    except:
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.5, **shade_kwargs)
    if labels is not None:
        ax.set(**labels)
    return ax.get_figure()


def plot_power_analysis(population_size, power, title=None, ax=None, **kwargs):
    if title is None:
        title = ""
    else:
        title = f"{title} "
    if ax is None:
        fig, ax = plt.subplots(figsize=get_fig_size(1, 1.2))
    ax.plot(population_size, power, **kwargs)
    ax.set_xlabel("Population size")
    ax.set_ylabel("Power")
    ax.set_title(title + "Power analysis")
    return ax.get_figure()


def plot_robertson(signal_levels, times, asd_correct_percent, nt_correct_percent, model, max_signal=0.158):
    asd_signals = np.argmax(asd_correct_percent > 0.8, axis=1)
    nt_signals = np.argmax(nt_correct_percent > 0.8, axis=1)

    fig = plt.figure(figsize=get_fig_size(1.5, 2))
    gs = GridSpec(2, 2, height_ratios=[2, 1], figure=fig)
    model_ax = fig.add_subplot(gs[0, :])
    ax = fig.add_subplot(gs[1, 0])
    robertson_ax = fig.add_subplot(gs[1, 1])
    real_ratios = plot_robertson_vs_lca(ax, robertson_ax, nt_signals, asd_signals, signal_levels, times,
                                        max_signal)

    print("Signal ratios: ", (signal_levels[asd_signals] / signal_levels[nt_signals]))
    print("real ratios: ", real_ratios)

    plot_lca(model_ax, model)
    # model.plot_trajectory(0, labels=[r"$x_1$", r"$x_2$"], ax=subax)

    return fig, []


def plot_robertson_vs_lca(ax, robertson_ax, nt_signals, asd_signals, signal_levels, times, max_signal,
                          ax_color=c.ASD_COLOR, labels=None):
    if labels is None:
        labels = ["NDR", "IDR"]
    if max_signal:
        signal_levels = 100 * ((0.12 * max_signal + signal_levels) / max_signal)
    ax.plot(np.arange(1, times.size + 1), np.round(signal_levels[nt_signals]), label=labels[0], marker='o',
            color=c.NT_COLOR)
    ax.plot(np.arange(1, times.size + 1), np.round(signal_levels[asd_signals]), label=labels[1], marker='o',
            color=ax_color)
    ax.legend()
    ax.set_xticks(np.arange(1, times.size + 1), np.round(100 * times).astype(int))
    ax.set_xlabel("Steps")
    ax.set_ylabel("% Signal level")
    ax.set_title("")
    ax.xaxis.set_tick_params(which='minor', length=0)
    real_ratios = np.array([37 / 23, 24 / 16, 16 / 15])
    if robertson_ax is not None:
        robertson_ax.errorbar([1, 2, 3], [23, 16, 15], yerr=[2, 2.2, 2.2], label="Controls", marker='o',
                              color="#2e3192")
        robertson_ax.errorbar([1, 2, 3], [37, 24, 16], yerr=[7, 5, 2], label="ASD", marker='o', color="#7e1a3b")
        robertson_ax.legend()
        robertson_ax.set_xticks([1, 2, 3], [200, 400, 1500])
        robertson_ax.set_xlabel("Stimulus viewing duration (ms)")
        robertson_ax.set_ylabel("%Coherent dots")
        robertson_ax.set_title("")
        robertson_ax.xaxis.set_tick_params(which='minor', length=0)
        if robertson_ax.get_ylim()[1] > max(signal_levels[asd_signals].max(), signal_levels[nt_signals].max()):
            robertson_ax.set_yticks([0, 10, 20, 30, 40])
            ax.set_ylim(0, robertson_ax.get_ylim()[1])
        else:
            robertson_ax.set_ylim(0, ax.get_ylim()[1])
            ax.set_ylim(0, ax.get_ylim()[1])
    ax.text(2.5, ax.get_ylim()[0], "  / /  ", backgroundcolor="white", va='center', ha='center')
    if robertson_ax is not None:
        robertson_ax.text(2.5, robertson_ax.get_ylim()[0], "  / /  ", backgroundcolor="white", va='center', ha='center')
    return real_ratios


def plot_lca(ax=None, model=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    ax: plt.Axes
    ax.grid('off')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    circle_linewidth = 2
    arrow_linewidth = 2
    excitation_marker_size = 30
    fontsize = 20
    title_y = 0.45
    radius = 0.08
    top_height = 0.85
    bottom_height = 0.65
    first_center = 0.1
    arrow_dx = radius * 1.3
    line_shift = (radius / 10)
    second_center = first_center + arrow_dx + 2 * radius
    third_center = second_center + arrow_dx + 2 * radius

    inhib_dx = np.cos(np.pi / 6) * radius
    inhib_dy = np.sin(np.pi / 6) * radius

    top_leak_x_tail = third_center + (np.cos((12 * np.pi) / 24) * radius)
    top_leak_x_head = third_center + (np.cos(2 * np.pi / 24) * radius)

    top_leak_y_tail = top_height + (np.sin((12 * np.pi) / 24) * radius)
    top_leak_y_head = top_height + (np.sin(2 * np.pi / 24) * radius)

    bottom_leak_x_tail = third_center + (np.cos((12 * np.pi) / 24) * radius)
    bottom_leak_x_head = third_center + (np.cos(2 * np.pi / 24) * radius)

    bottom_leak_y_tail = bottom_height - (np.sin((12 * np.pi) / 24) * radius)
    bottom_leak_y_head = bottom_height - (np.sin(2 * np.pi / 24) * radius)
    input_circle = plt.Circle((first_center, (top_height + bottom_height) / 2), radius, color='k', fill=False,
                              linewidth=circle_linewidth)

    ax.text(first_center, (top_height + bottom_height) / 2, r"$I\in[0,0.5]$", ha='center', va='center',
            fontsize=fontsize * 0.8)
    ax.plot([first_center + radius, first_center + radius + arrow_dx - line_shift],
            [(top_height + bottom_height) / 2, top_height],
            color=c.ASD_COLOR, linewidth=arrow_linewidth)
    ax.plot([first_center + radius, first_center + radius + arrow_dx - line_shift],
            [(top_height + bottom_height) / 2, bottom_height],
            color=c.ASD_COLOR, linewidth=arrow_linewidth)
    ax.scatter([first_center + radius + arrow_dx - line_shift] * 2, [top_height, bottom_height], excitation_marker_size,
               color=c.ASD_COLOR)

    enc1 = plt.Circle((second_center, top_height), radius, color='k', fill=False, linewidth=circle_linewidth)
    ax.add_patch(enc1)
    ax.text(second_center, top_height, '$f(I)$', ha='center', va='center', fontsize=fontsize)
    enc2 = plt.Circle((second_center, bottom_height), radius, color='k', fill=False, linewidth=circle_linewidth)
    ax.text(second_center, bottom_height, '$1-f(I)$', ha='center', va='center', fontsize=fontsize)

    ax.hlines([top_height, bottom_height], second_center + radius,
              second_center + radius + arrow_dx - line_shift, color=c.ASD_COLOR, linewidth=arrow_linewidth)
    ax.scatter([second_center + radius + arrow_dx - line_shift] * 2, [top_height, bottom_height],
               excitation_marker_size,
               color=c.ASD_COLOR)

    pop1 = plt.Circle((third_center, top_height), radius, color='k', fill=False, linewidth=circle_linewidth)
    ax.text(third_center, top_height, '$y_1$', ha='center', va='center', fontsize=fontsize)

    pop2 = plt.Circle((third_center, bottom_height), radius, color='k', fill=False, linewidth=circle_linewidth)
    ax.text(third_center, bottom_height, '$y_2$', ha='center', va='center', fontsize=fontsize)

    top_out_inhib = FancyArrowPatch((third_center - inhib_dx, top_height - inhib_dy),
                                    (third_center - inhib_dx - line_shift, bottom_height + inhib_dy + line_shift),
                                    arrowstyle='|-|,widthA=0,widthB=8,angleB=20', connectionstyle='arc3,rad=0.3',
                                    color=c.EI_COLOR,
                                    linewidth=arrow_linewidth, shrinkA=0, shrinkB=0)

    bottom_out_inhib = FancyArrowPatch((third_center + inhib_dx, bottom_height + inhib_dy),
                                       (third_center + inhib_dx + line_shift, top_height - inhib_dy - line_shift),
                                       arrowstyle='|-|,widthA=0,widthB=8,angleB=20', connectionstyle='arc3,rad=0.3',
                                       color=c.EI_COLOR,
                                       linewidth=arrow_linewidth, shrinkA=0, shrinkB=0)
    ax.text(third_center - inhib_dx * 1.1, (top_height + bottom_height) / 2, r"$\beta_{2,1}$", ha='left', va='center',
            fontsize=fontsize)
    ax.text(third_center + inhib_dx * 1.1, (top_height + bottom_height) / 2, r"$\beta_{1,2}$", ha='right', va='center',
            fontsize=fontsize)

    top_leak = FancyArrowPatch((top_leak_x_tail, top_leak_y_tail),
                               (top_leak_x_head - line_shift, top_leak_y_head + 4 * line_shift),
                               arrowstyle='|-|,widthA=0,angleA=30,angleB=-50,widthB=8', connectionstyle=f'arc3,rad=-1',
                               color=c.EI_COLOR, linewidth=arrow_linewidth, shrinkA=0, shrinkB=0)

    bottom_leak = FancyArrowPatch((bottom_leak_x_tail, bottom_leak_y_tail),
                                  (bottom_leak_x_head - line_shift, bottom_leak_y_head - 4 * line_shift),
                                  arrowstyle='|-|,widthA=0,angleA=30,angleB=50,widthB=8', connectionstyle=f'arc3,rad=1',
                                  color=c.EI_COLOR, linewidth=arrow_linewidth, shrinkA=0, shrinkB=0)

    ax.text(third_center + radius, top_height + 1.25 * radius, r"$\kappa_2$", ha='center', va='bottom',
            fontsize=fontsize)
    ax.text(third_center + radius, bottom_height - 1.25 * radius, r"$\kappa_1$", ha='center', va='top',
            fontsize=fontsize)
    top_arrow = ax.arrow(third_center + radius, top_height, 1.7 * arrow_dx, 0, head_width=0.02, head_length=0.01,
                         length_includes_head=True, color=c.HIST_COLOR, linewidth=arrow_linewidth)
    bottom_arrow = ax.arrow(third_center + radius, bottom_height, 1.15 * arrow_dx, 0, head_width=0.02, head_length=0.01,
                            length_includes_head=True, color=c.EI_COLOR, linewidth=arrow_linewidth)
    ax.vlines(third_center + radius + 1.55 * arrow_dx, bottom_height * 0.9, top_height * 1.1, linestyles=":",
              colors="k", linewidth=arrow_linewidth * 2)

    ax.text(first_center, title_y, "Motion Coherence", va='center', ha='center', fontweight='bold', fontsize=fontsize)
    ax.text(second_center, title_y, "Encoding", va='center', ha='center', fontweight='bold', fontsize=fontsize)
    ax.text(third_center, title_y, "Accumulator", va='center', ha='center', fontweight='bold', fontsize=fontsize)
    ax.text(third_center + radius + 1.55 * arrow_dx, title_y, "Decision", va='center', ha='center', fontweight='bold',
            fontsize=fontsize)
    ax.add_patch(input_circle)
    ax.add_patch(enc2)
    ax.add_patch(pop1)
    ax.add_patch(pop2)
    ax.add_patch(top_out_inhib)
    ax.add_patch(bottom_out_inhib)
    ax.add_patch(top_leak)
    ax.add_patch(bottom_leak)
    ax.add_patch(top_arrow)
    ax.add_patch(bottom_arrow)

    start_x = 0.015
    ax_y = 0.13
    width = 15
    height = 25
    hspace = 0.115
    ax_stim = inset_axes(ax, width=f"{width}%", height=f"{height}%", loc='lower left',
                         bbox_to_anchor=(start_x, ax_y, 1, 1),
                         bbox_transform=ax.transAxes)
    ax_stim.imshow(plt.imread("data/stim_image.png"))
    ax_encoding = inset_axes(ax, width=f"{width}%", height=f"{1.5 * height}%", loc='lower left',
                             bbox_to_anchor=(start_x + (width / 100) + hspace, 0.5 * ax_y, 1, 1),
                             bbox_transform=ax.transAxes)
    ax_encoding.imshow(plt.imread("data/Encoding figure.png"))
    ax_accumulation = inset_axes(ax, width=f"{width}%", height=f"{height}%", loc='lower left',
                                 bbox_to_anchor=(start_x + 2 * ((width / 100) + hspace), ax_y, 1, 1),
                                 bbox_transform=ax.transAxes)
    model.plot_trajectory(0, labels=False, ax=ax_accumulation, colors=[c.EI_COLOR, c.HIST_COLOR])

    ax_decision = inset_axes(ax, width=f"{width}%", height=f"{1.5 * height}%", loc='lower left',
                             bbox_to_anchor=(start_x + 3 * ((width / 100) + hspace), 0.9 * ax_y, 1, 1),
                             bbox_transform=ax.transAxes)
    ax_decision.imshow(plt.imread("data/button press.png"))

    insets = [ax_stim, ax_encoding, ax_accumulation, ax_decision]
    for inset in insets:
        inset.axis('off')
    return ax, insets


def plot_combined_power_analysis(heterogeneities, nt_n, asd_n, population, powers, ax=None, labels=True):
    closest_to_80 = np.argmax(powers > 0.8, axis=-1)
    real_80_idxs = (powers > 0.8).any(-1)
    pops = [population[closest_to_80[i]][real_80_idxs[i]] for i in range(powers.shape[0])]
    hets = [heterogeneities[real_80_idxs[i]] for i in range(powers.shape[0])]

    if ax is None:
        fig, ax = plt.subplots(figsize=get_fig_size(1, 1.3))
    if labels:
        ax.set_xlabel("Heterogeneity between individuals")
        ax.set_ylabel("Single group size")
    labels = ["Binocular rivalry", "Encoding capacity", "Slow updating", "Motion coherence", "LCA 1500 steps"]
    for i in range(len(pops) - 1):
        ax.plot(hets[i], pops[i], label=labels[i])
    ax.legend()
    asd_hets, asd_ns, nt_hets, nt_ns = utils.get_tapping_heterogeneities()
    ax.axvline((asd_hets.std() + nt_hets.std()) / 2, color='gray', linestyle='--')
    return ax


def plot_effect_of_sampling_on_dynamic_range(ax=None):
    _ = utils.get_effective_n_from_heterogeneity(0.1)
    noise_level_list, retrieved_n = utils.HETEROGENEITY_TO_N
    _, retrieved_n_std = utils.HETEROGENEITY_TO_N_STD
    # plot
    if ax is None:
        fig = plt.figure(figsize=get_fig_size(1, 1.7))
        ax: plt.Axes = fig.subplots()
    ax.plot(noise_level_list, retrieved_n, color=c.HIST_COLOR, linewidth=0.8)
    ax.fill_between(noise_level_list, retrieved_n - retrieved_n_std, retrieved_n + retrieved_n_std, color=c.HIST_COLOR,
                    alpha=0.3)
    axline_n_value = 10
    n_width = 2
    centers = [[9, 11], [8, 12], [7, 13]]
    axline_idx = retrieved_n.size - np.searchsorted(retrieved_n[::-1], axline_n_value)

    def get_matching_rects_data(centers, n_width):
        above = [centers[1] + n_width / 2, centers[1] - n_width / 2]
        below = [centers[0] + n_width / 2, centers[0] - n_width / 2]
        above_idx = retrieved_n.size - np.searchsorted(retrieved_n[::-1], above)
        below_idx = retrieved_n.size - np.searchsorted(retrieved_n[::-1], below)
        return above, above_idx, below, below_idx

    linestyles = ["-", "--", "-."]
    colors = ["#F98400", "#7294d4", "#D8A499"]
    n_blocks = 3
    # start_height = 3
    for idx in range(n_blocks):
        above, above_idx, below, below_idx = get_matching_rects_data(centers[idx], n_width)
        above_rect = plt.Rectangle((noise_level_list[above_idx[0]], retrieved_n[above_idx[0]]),
                                   np.diff(noise_level_list[above_idx]).item(),
                                   np.diff(retrieved_n[above_idx]).item(),
                                   linewidth=3, fill=False, edgecolor=colors[idx], alpha=0.7,
                                   linestyle=linestyles[idx])
        below_rect = plt.Rectangle((noise_level_list[below_idx[0]], retrieved_n[below_idx[0]]),
                                   np.diff(noise_level_list[below_idx]).item(),
                                   np.diff(retrieved_n[below_idx]).item(),
                                   linewidth=3, fill=False, edgecolor=colors[idx], alpha=0.7,
                                   linestyle=linestyles[idx])

        ax.add_patch(above_rect)
        ax.add_patch(below_rect)
    ax.axvline(noise_level_list[axline_idx], color='k', linestyle='--')
    ax.set_xlabel("Neural population heterogeneity")
    ax.set_ylabel("Hill-coefficient (n)")
    asd_hets, asd_ns, nt_hets, nt_ns = utils.get_tapping_heterogeneities()
    asd_y = 2
    nt_y = 3
    asd_mean_het = utils.get_heterogeneity_from_n(asd_ns.mean(), 20)
    nt_mean_het = utils.get_heterogeneity_from_n(nt_ns.mean(), 20)
    ax.scatter(asd_mean_het, asd_y, color=c.DARK_ASD_COLOR, s=25)
    ax.hlines(asd_y, np.quantile(asd_hets, 0.25), np.quantile(asd_hets, 0.75), color=c.DARK_ASD_COLOR, linewidths=1.6)
    ax.scatter(nt_mean_het, nt_y, color=c.DARK_NT_COLOR, s=25)
    ax.hlines(nt_y, np.quantile(nt_hets, 0.25), np.quantile(nt_hets, 0.75), color=c.DARK_NT_COLOR, linewidths=1.6)
    asd_legend = plt.Line2D([], [], color=c.DARK_ASD_COLOR, marker='o', markersize=7, label='ASD tapping fit')
    nt_legend = plt.Line2D([], [], color=c.DARK_NT_COLOR, marker='o', markersize=7, label='NT tapping fit')
    ax.legend(handles=[asd_legend, nt_legend])
    ax.set_xlim(0, 0.3)


def plot_power_analysis_figure(populations, powers, heterogeneities, nt_n, asd_n):
    fig: plt.Figure = plt.figure(figsize=get_fig_size(1.25, 1.35))
    gs = GridSpec(5, len(nt_n), fig, height_ratios=[0.000001, 1, 0.000001, 1, 0.000001])
    axes = [fig.add_subplot(gs[1, :]), fig.add_subplot(gs[-2, 0]), fig.add_subplot(gs[-2, 1])]
    plot_effect_of_sampling_on_dynamic_range(axes[0])
    for i in range(len(nt_n)):
        plot_combined_power_analysis(heterogeneities[i], nt_n[i], asd_n[i], populations[i], powers[i], axes[i + 1],
                                     labels=False)

    fig.text(0.5, 0.05, "Variability between individuals", ha='center', va='center', fontsize=20, fontweight='bold')
    axes[1].set_ylabel("Single group size")
    return fig


def plot_hgf_param_trajectory(r, ax, title):
    t = np.ones_like(r.u)
    ts = np.concatenate([[0], np.cumsum(t)])
    ax.plot(ts, np.concatenate([[utils.tapas_sgm(r.p_prc.mu_0[1], 1)], utils.tapas_sgm(r.traj.mu[:, 1], 1)]),
            color='r', linewidth=4, label=r"$P(x_1=1)$")
    ax.scatter(0, utils.tapas_sgm(r.p_prc.mu_0[1], 1), color="red", s=15)
    ax.plot(ts[1:np.argmax(r.u == 1) + 1], r.u[:np.argmax(r.u == 1)], color=[0, 0.6, 0], label="$u$, Input",
            linewidth=3,
            linestyle=":")
    ax.plot(ts[np.argmax(r.u == 1) + 1:], r.u[np.argmax(r.u == 1):], color=[0, 0.6, 0], linewidth=3, linestyle=":")
    ax.plot(ts[1:np.argmax(r.y == 1) + 1], (((r.y - 0.5) * 1.05) + 0.5)[:np.argmax(r.y == 1)], color=[1, 0.7, 0],
            label="$y$, Response", linewidth=3, linestyle=":")
    ax.plot(ts[np.argmax(r.y == 1) + 1:], (((r.y - 0.5) * 1.05) + 0.5)[np.argmax(r.y == 1):], color=[1, 0.7, 0],
            linewidth=3, linestyle=":")
    ax.plot(ts[1:], r.traj.wt[:, 0], color='k', linestyle=":", linewidth=3, label="Learning rate")
    ax.legend(loc='upper left', bbox_to_anchor=[0, 0.9, 0.2, 0.1], fontsize=15)
    ax.set_title(title, fontsize=25)
    ax.set_ylabel("Input and Response", fontsize=20)
    ax.set_xlabel("Trial number", fontsize=20)
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='y', which='major', labelsize=17)
    ax.tick_params(axis='y', which='minor', labelsize=17)


def plot_hgf_boxplots(arr, ax, title, ylabel):
    ax.boxplot(arr[:, 0], positions=[0.5], showfliers=False, widths=0.4,
               capprops={"color": c.ASD_COLOR, "alpha": 0.7},
               medianprops={"color": c.ASD_COLOR, "linewidth": 3},
               whiskerprops=dict(color=c.ASD_COLOR, alpha=0.7, linewidth=2),
               boxprops=dict(color=c.ASD_COLOR))
    ax.scatter(np.full_like(arr[:, 0], 0.5), arr[:, 0], c="none", s=15, edgecolor='k')
    ax.boxplot(arr[:, 1], positions=[1], showfliers=False, widths=0.4,
               capprops={"color": c.NT_COLOR, "alpha": 0.7},
               medianprops={"color": c.NT_COLOR, "linewidth": 3},
               whiskerprops=dict(color=c.NT_COLOR, alpha=0.7, linewidth=2),
               boxprops=dict(color=c.NT_COLOR))
    ax.scatter(np.full_like(arr[:, 0], 1), arr[:, 1], c="none", s=15, edgecolor='k')
    ax.set_title(title, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_xticklabels(["High PU", "Low PU"], fontweight='bold')
    ax.tick_params(axis='y', which='major', labelsize=17)
    ax.tick_params(axis='y', which='minor', labelsize=17)


def plot_hgf_trajectories(sim_fit, asd_fit, ax1, ax2):
    plot_hgf_param_trajectory(sim_fit, ax1, "Base parameters fit")
    plot_hgf_param_trajectory(asd_fit, ax2, "High PU parameters fit")


def plot_hgf(sim_fit, asd_fit, alphas, omegas):
    fig, axes = plt.subplots(2, 2, figsize=get_fig_size(1, 1.6))
    axes = np.ravel(axes)
    plot_hgf_trajectories(sim_fit, asd_fit, axes[0], axes[1])
    plot_hgf_parameters(alphas, omegas, axes[2], axes[3])
    return fig


def plot_hgf_parameters(alphas, omegas, ax1, ax2):
    plot_hgf_boxplots(alphas, ax1, "Perceptual uncertainty", r"$\alpha$")
    ax1.set_ylim([0, ax1.get_ylim()[1]])
    plot_hgf_boxplots(omegas, ax2, "Volatility estimates", r"$\omega_3$")
