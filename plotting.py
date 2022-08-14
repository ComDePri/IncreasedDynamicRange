import matplotlib.pyplot as plt
import numpy as np
from typing import List
from itertools import combinations
import constants as c
import wes
import os
import string
from matplotlib import rcParams, rc, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from skimage.color import rgb2lab, lab2rgb
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import utils

# parameter setting
rc('font', **{'family': 'DeJavu Serif', 'sans-serif': ['Helvetica']})
wes.set_palette('Darjeeling1')
rcParams.update({'axes.titleweight': 'bold',
                 'axes.labelweight': 'bold',
                 'axes.labelsize': c.AX_LABEL_SIZE,
                 'axes.titlesize': c.AX_TITLE_SIZE,
                 'axes.linewidth': c.LINEWIDTH,
                 'xtick.labelsize': c.TICK_LABELSIZE,
                 'ytick.labelsize': c.TICK_LABELSIZE,
                 'figure.titlesize': c.FIG_TITLE_SIZE,
                 'legend.shadow': False,
                 'legend.framealpha': c.FRAME_ALPHA,
                 'legend.fontsize': c.LEGEND_FONT_SIZE,
                 'figure.figsize': [c.SUBPLOT_SIZE * 1.3, c.SUBPLOT_SIZE * 1.3],
                 'figure.dpi': c.DPI,
                 'figure.subplot.left': c.LEFT_SPACE,
                 'figure.subplot.right': c.RIGHT_SPACE,
                 'figure.subplot.bottom': c.BOTTOM_SPACE,
                 'figure.subplot.top': c.TOP_SPACE,
                 'figure.subplot.wspace': c.WSPACE,
                 'figure.subplot.hspace': c.HSPACE,
                 'figure.frameon': False
                 })


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
            if x:
                axis.set_xscale("log")
            if y:
                axis.set_yscale("log")
    except TypeError:
        if x:
            ax.set_xscale("log")
        if y:
            ax.set_yscale("log")


def savefig(fig: plt.Figure, save_name: str, shift_x=-0.1, shift_y=1.01, ignore: [None, list] = None, si=False,
            tight=True, numbering_size=40) -> None:
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
    if save_name[-4:].lower() != ".png":
        save_name += ".png"
    if tight:
        fig.tight_layout()
    if len(fig.axes) - len(ignore) > 1:
        for i, ax in enumerate(fig.axes):
            if ax in ignore or ax.get_label() == '<colorbar>':
                continue
            ax.text(shift_x, shift_y, string.ascii_uppercase[i], transform=ax.transAxes,
                    size=numbering_size, weight='bold')
    fig.savefig(os.path.join(c.SI_FIG_DIR if si else c.FIG_DIR, save_name), pad_inches=0.15)


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
    if len(boxplot_value_list[0].shape)>1:
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
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(23)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(23)
    return fig_population_resp


def plot_variance_over_signal_range(signal, nt_variance, asd_variance, ei_variance):
    fig = plt.figure()
    ax: plt.Axes = fig.subplots()
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.05)
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    set_ax_labels(ax,
                  "",
                  "Signal level", "Variance in population responses", ax_label_size=27)
    ax.scatter(signal, nt_variance, s=2, alpha=0.5, c=c.NT_COLOR, label="NT")
    ax.text(0.32, 5e-4, "NT", color=c.NT_COLOR, size=25, weight="bold")
    ax.scatter(signal, asd_variance, s=2, alpha=0.5, c=c.ASD_COLOR, label="ASD")
    ax.text(0.22, 2.9e-4, "ASD", color=c.ASD_COLOR, size=25, weight="bold")
    ax.scatter(signal, ei_variance, s=2, alpha=0.5, c=c.EI_COLOR, label="E/I")
    ax.text(0.65, 4.5e-4, "E/I", color=c.EI_COLOR, size=25, weight="bold")
    return fig


def plot_km_tracking(ax: plt.Axes, km, time):
    # ax.plot(time, km[:, :, 0].T, color=c.NT_COLOR, zorder=1, alpha=0.3, linewidth=0.6)
    # ax.plot([], [], color=c.NT_COLOR, label="NT")
    # ax.plot(time, km[:, :, 1].T, color=c.ASD_COLOR, zorder=2, alpha=0.3, linewidth=0.6)
    # ax.plot([], [], color=c.ASD_COLOR, label="ASD")
    ax.plot(time, km[:, :, 0].mean(0), color=c.NT_COLOR, zorder=1, linewidth=1, label="NT")
    ax.fill_between(time, km[:, :, 0].mean(0) + km[:, :, 0].std(0), km[:, :, 0].mean(0) - km[:, :, 0].std(0), alpha=0.3,
                    color=c.NT_COLOR)
    ax.plot(time, km[:, :, 1].mean(0), color=c.ASD_COLOR, zorder=2, linewidth=1, label="ASD")
    ax.fill_between(time, km[:, :, 1].mean(0) + km[:, :, 1].std(0), km[:, :, 1].mean(0) - km[:, :, 1].std(0), alpha=0.3,
                    color=c.ASD_COLOR)
    ax.hlines(c.LR_THRESHOLD, 0, c.LR_MAX_T, linestyles=":", colors="black", label="threshold", zorder=5)
    # ax.legend()


def variance_subplot(ax, asd_cv, nt_cv):
    subax: plt.Axes = inset_axes(ax, width="70%", height="60%", loc="lower right")
    subax.plot(nt_cv, color=c.NT_COLOR, zorder=1, label="NT")
    subax.plot(asd_cv, color=c.ASD_COLOR, zorder=2, label="ASD")
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
    ax.hist(nt_lr, color=c.NT_COLOR, alpha=0.7, label="NT learning rate", bins=bins)
    ax.hist(asd_lr, color=c.ASD_COLOR, alpha=0.7, label="ASD learning rate", bins=bins)
    ax.legend()


def plot_last_km_distributions(ax, km):
    bins = np.linspace(km[:, -1, :].min(), km[-1, :, :].max(), c.LR_LAST_KM_BIN_NUM)
    ax.hist(km[:, -1, 0], bins=bins, color=c.NT_COLOR, alpha=0.7, label="NT")
    ax.hist(km[:, -1, 1], bins=bins, color=c.ASD_COLOR, alpha=0.7, label="ASD")
    ax.vlines(c.LR_THRESHOLD, 0, km[:, -1, :].max(), colors="k", linestyles=":")


# old, with 3 figures
# def plot_learning_rate_and_accuracy(km, asd_cv, nt_cv, asd_lr, nt_lr) -> [plt.Figure, plt.Axes]:
#     lr_fig, axes = get_3_axes_with_3rd_centered()
#     set_labels(axes,
#                ["Increased dynamic range reduces accuracy and\nincreases variance in the learning process",
#                 "Increased dynamic range induces slow updating",
#                 "Increased dynamic range reduces accuracy and\nincreases variance in the final result"],
#                ["Time (steps)", "Learning rate", r"Learned half-point"],
#                [r"$\hat{K_m}$", "Frequency", "Frequency"])
#     plot_km_tracking(axes[0], km)
#     subax = variance_subplot(axes[0], asd_cv, nt_cv)
#     plot_learning_rate(axes[1], asd_lr, nt_lr)
#     plot_last_km_distributions(axes[2], km)
#     return lr_fig, subax

def plot_learning_rate_and_accuracy(km, time, cluster_df,
                                    nt_line_text=(500, 0.45), asd_line_text=(1350, 0.425),
                                    nt_hist_text=(0.49, 15.5), asd_hist_text=(0.475, 14)) -> [plt.Figure, plt.Axes]:
    lr_fig = plt.figure()
    axes = lr_fig.subplots()
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.07)
    set_labels(axes, "", "Time (steps)", r"Learned value")
    plot_km_tracking(axes, km, time)
    if cluster_df is not None:
        significant_cluster_df: pd.DataFrame = cluster_df.loc[cluster_df['p-value'] < 0.05,]
        for i, row in significant_cluster_df.iterrows():
            axes.axvspan(row['Start index'], row['End index'], alpha=0.2, color=c.EI_COLOR)
    subax: plt.Axes = inset_axes(axes, width="100%", height="100%", loc="lower left", bbox_transform=axes.transAxes,
                                 bbox_to_anchor=(0.3, 0.06, 0.65, 0.6))
    plot_last_km_distributions(subax, km)
    subax.set_title("Last learnt value", fontsize=30)
    subax.set_ylabel(f"Frequency", fontsize=30)
    axes.text(nt_line_text[0], nt_line_text[1], "NT", color=c.NT_COLOR, size=30, weight="bold")
    axes.text(asd_line_text[0], asd_line_text[1], "ASD", color=c.ASD_COLOR, size=30, weight="bold")
    subax.text(nt_hist_text[0], nt_hist_text[1], "NT", color=c.NT_COLOR, size=30, weight="bold")
    subax.text(asd_hist_text[0], asd_hist_text[1], "ASD", color=c.ASD_COLOR, size=30, weight="bold")
    axes.tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    subax.tick_params(axis='both', which='major', labelsize=25, width=3, length=15)

    return lr_fig, subax


def plot_sensitivity_to_signal_differences(s, nt_resp, asd_resp, si=False):
    fig = plt.figure()
    ax: plt.Axes = fig.subplots()
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.1)
    ax.plot(s, nt_resp, color=c.NT_COLOR, label=f"Sharp, n={c.SI_SSD_NT_N if si else c.SSD_NT_N}", zorder=1,
            linewidth=3)
    ax.plot(s, asd_resp, color=c.ASD_COLOR, label=f"Gradual, n={c.SI_SSD_ASD_N if si else c.SSD_ASD_N}", zorder=2,
            linewidth=3)

    ax.legend(fontsize=25)
    nt_low_diff = np.abs(nt_resp[c.SSD_LOW_SIG2_IDX] - nt_resp[c.SSD_LOW_SIG1_IDX])
    asd_low_diff = np.abs(asd_resp[c.SSD_LOW_SIG2_IDX] - asd_resp[c.SSD_LOW_SIG1_IDX])
    nt_high_diff = np.abs(nt_resp[c.SSD_HIGH_SIG2_IDX] - nt_resp[c.SSD_HIGH_SIG1_IDX])
    asd_high_diff = np.abs(asd_resp[c.SSD_HIGH_SIG2_IDX] - asd_resp[c.SSD_HIGH_SIG1_IDX])
    sig_idx_array = np.array([c.SSD_LOW_SIG1_IDX, c.SSD_LOW_SIG2_IDX, c.SSD_HIGH_SIG1_IDX, c.SSD_HIGH_SIG2_IDX])
    ax.vlines(s[sig_idx_array[:2]], -.1, asd_resp[sig_idx_array[:2]], linestyles=":", colors="black", zorder=3)
    ax.vlines(s[sig_idx_array[2:]], -.1, nt_resp[sig_idx_array[2:]], linestyles=":", colors="black", zorder=4)
    set_ax_labels(ax, "", "Signal level", r"Neural gain")
    ax.text(0.81, 0.65, "$\\Delta A_{NT}=%s$\n$\\Delta A_{ASD}=%s$" % (
        str(utils.num2print(nt_high_diff)).replace("-", "^{-") + "}",
        str(utils.num2print(asd_high_diff)).replace("-", "^{-") + "}"),
            fontsize=20)
    ax.text(0, 0.05, "$\\Delta A_{NT}=%s$\n$\\Delta A_{ASD}=%s$" % (
        str(utils.num2print(nt_low_diff)).replace("-", "^{-") + "}",
        str(utils.num2print(asd_low_diff)).replace("-", "^{-") + "}"),
            fontsize=20)
    ax.text(0.6, 0.69, "ASD", color=c.ASD_COLOR, fontsize=24, weight='bold')
    ax.text(0.435, 0.82, "NT", color=c.NT_COLOR, fontsize=24, weight='bold')
    ax.text(0.15, -0.05, "A", fontsize=20, fontweight='bold')
    ax.text(0.32, -0.05, "B", fontsize=20, fontweight='bold')
    ax.text(0.65, -0.05, "C", fontsize=20, fontweight='bold')
    ax.text(0.82, -0.05, "D", fontsize=20, fontweight='bold')
    return fig


def plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate, text_x=(140, 25), text_y=(100, 300)) -> plt.Figure:
    fig = plt.figure(figsize=get_fig_size(1, 1.2))
    axs = fig.subplots()
    plt.subplots_adjust(left=c.LEFT_SPACE + 0.05)
    change_timepoint = round(max(c.SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
    asd_times = np.argmax(np.abs(asd_sig_estimate) >= c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX, axis=1) - change_timepoint
    nt_times = np.argmax(np.abs(nt_sig_estimate) >= c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX, axis=1) - change_timepoint
    min_bin = min(asd_times.min(), nt_times.min())
    max_bin = max(asd_times.max(), nt_times.max())
    bins = np.linspace(min_bin, max_bin, 25)

    axs.plot(asd_sig_estimate[c.SR_PLOT_REP_IDX, :], label="ASD", color=c.ASD_COLOR, linewidth=3)
    axs.plot(nt_sig_estimate[c.SR_PLOT_REP_IDX, :], label="NT", color=c.NT_COLOR, linewidth=2, alpha=0.7)
    axs.plot(signal[c.SR_PLOT_REP_IDX, :, 0], color='gray', label="measured signal", alpha=0.3)
    axs.plot([c.SR_SIG_MIN] * change_timepoint +
             [c.SR_SIG_MAX] * (c.SR_NUM_STEPS - change_timepoint), color='k',
             label="real signal", linewidth=3, zorder=1)

    subax = inset_axes(axs, "50%", "50%", loc="lower right", borderpad=3, bbox_transform=axs.transAxes,
                       bbox_to_anchor=(0.05, 0.05, 1, 1))  # type: plt.Axes
    subax.hist(asd_times, bins=bins, label="ASD")
    subax.hist(nt_times, bins=bins, label="NT")
    subax.text(text_x[0], text_y[0], "ASD", color=c.ASD_COLOR, size=25, weight="bold")
    subax.text(text_x[1], text_y[1], "NT", color=c.NT_COLOR, size=25, weight="bold")

    set_labels(fig.get_axes(), "",
               ["Time (steps)", "Time after switch (steps)"],
               ["Signal", "Frequency"])
    subax.xaxis.set_label_text("Time after switch (steps)", fontsize=20)
    subax.yaxis.set_label_text("Frequency", fontsize=20)

    return fig


def plot_binocular_rivalry(asd_transition_count, nt_transition_count, asd_ratios, nt_ratios, asd_resp, nt_resp,
                           signal, example_idx=0) -> plt.Figure:
    fig, axes = get_3_axes_with_1st_full()  # fig.subplots(2, 1)
    fig.subplots_adjust(hspace=c.HSPACE + 1, wspace=c.WSPACE + 0.2, left=c.LEFT_SPACE + 0.01,
                        bottom=c.BOTTOM_SPACE - 0.02)
    max_transitions = max(np.max(nt_transition_count), np.max(asd_transition_count))
    max_ratios = 50
    axes[0].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    axes[1].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    axes[2].tick_params(axis='both', which='major', labelsize=25, width=5, length=20)

    axes[1].hist(asd_transition_count, color=c.ASD_COLOR, bins=np.arange(0, max_transitions, 2), label="ASD", alpha=0.7)
    axes[1].hist(nt_transition_count, color=c.NT_COLOR, bins=np.arange(0, max_transitions, 2), label="NT", alpha=0.7)
    axes[1].text(5, 150, "ASD", color=c.ASD_COLOR, fontsize=30, weight='bold')
    axes[1].text(15, 60, "NT", color=c.NT_COLOR, fontsize=30, weight='bold')
    # axes[0].legend()
    axes[2].hist(asd_ratios, label="ASD", bins=np.arange(0, max_ratios, .5), alpha=0.7)
    axes[2].hist(nt_ratios, label="NT", bins=np.arange(0, max_ratios, .5), alpha=0.7)
    # axes[2].legend()
    axes[2].text(4, 70, "ASD", color=c.ASD_COLOR, fontsize=30, weight='bold')
    axes[2].text(15, 17, "NT", color=c.NT_COLOR, fontsize=30, weight='bold')
    axes[0].plot(signal[example_idx, :, 0], linewidth=4, color='k', label="Signal")
    axes[0].plot(asd_resp[example_idx, :], alpha=0.5, linewidth=4, color=c.ASD_COLOR, label="Gradual, ASD")
    axes[0].plot(nt_resp[example_idx, :], alpha=0.5, linewidth=4, color=c.NT_COLOR, label="Sharp, NT")
    axes[0].hlines([0.8, 0.2], xmin=0, xmax=nt_resp[example_idx, :].size, alpha=0.5, linewidth=2, linestyle=":",
                   color='k')
    axes[2].legend(fontsize=30)
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
    color_list = get_normed_color_mix(c.ASD_COLOR, c.EI_COLOR, effective_n)
    if ax is None:
        fig, ax = plt.subplots()  # pl.get_3_axes_with_3rd_centered()
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


def plot_fi(fi, s, si=False):
    fi_n_list = c.SI_FI_N_LIST if si else c.FI_N_LIST
    s_dt = s[1] - s[0]
    fig = plt.figure()
    fig.subplots_adjust(left=c.LEFT_SPACE + 0.08)
    mix_colors = get_normed_color_mix(c.ASD_COLOR, c.NT_COLOR, fi_n_list)
    ax = fig.subplots()
    subax: plt.Axes = inset_axes(ax, width="30%", height="30%")
    subax.set_xlabel("Hill-coefficient\nn", fontsize=23)
    subax.set_ylabel("$\int_0^1\sqrt{I_F(S)}dS$", fontsize=18)
    total_enc_capacity = []
    for i, enc_cap in enumerate(fi):
        ax.plot(s, np.sqrt(enc_cap), color=mix_colors[i], linewidth=4,
                label="$n=%d$" % (fi_n_list[i]))
        total_enc_capacity.append(enc_cap.sum() * s_dt)
    subax.bar(fi_n_list, total_enc_capacity, color=mix_colors)
    set_ax_labels(ax, "", "Signal level", "Encoding capacity $\\sqrt{I_F(S)}$", title_size=30, ax_label_size=30)
    ax.legend(fontsize=25, loc="upper left")
    ax.tick_params(axis='both', which='major', labelsize=25, width=5, length=20)
    subax.tick_params(axis='both', which='major', width=3, length=15)
    return fig


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
    fig, ax = plt.subplots(figsize=get_fig_size(1, 1.3))
    set_ax_labels(ax, "Time to 95% of signal after change as a function of slope (n)",
                  "n", "Time to 95% of signal after change", 0.75 * c.AX_TITLE_SIZE, 0.75 * c.AX_LABEL_SIZE)
    ax.plot(n_range, time_to_track, '-o')
    make_log_scale(ax)
    return fig


def plot_dynamic_range_as_function_of_slope(slopes, widths, ratios):
    fig, axes = plt.subplots(1, 2, figsize=get_fig_size(1, 2.2))
    axes[0].set(title="Response width VS slope (n)", xlabel="n", ylabel=r"$S_{0.9}-S_{0.1}$")
    axes[0].plot(slopes, widths, '-o')
    axes[1].set(title="Dynamic range ratio (R) VS slope (n)", xlabel="n", ylabel=r"$\frac{S_{0.9}}{S_{0.1}}$")
    axes[1].plot(slopes, ratios, '-o')
    make_log_scale(axes)
    return fig
