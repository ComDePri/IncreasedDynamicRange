import numpy as np
import scipy.stats
from tqdm import tqdm

import constants as c
import idr_utils as utils
import plotting as pl

pl.plt.style.use('comdepri.mplstyle')


def variance_simulation(si=False):
    print("======================================\n"
          "========= Variance Simulation ========\n"
          "======================================")
    np.random.seed(c.SEED)
    asd_variance, ei_variance, nt_variance, signal = get_variance(si)
    # calculate CI of .5 and 1/e from max variance
    width_ci_high, width_ci_low, widths = get_bootstrapped_widths(asd_variance, ei_variance, nt_variance)
    print(
        f"ASD variance width (max/2): {widths[0][0]:.4g}, 99% CI: [{width_ci_low[0][0]:.4g}, {width_ci_high[0][0]:.4g}]\n"
        f"E\I variance width (max/2): {widths[0][1]:.4g}, 99% CI: [{width_ci_low[0][1]:.4g}, {width_ci_high[0][1]:.4g}]\n"
        f"NT variance width (max/2): {widths[0][2]:.4g}, 99% CI: [{width_ci_low[0][2]:.4g}, {width_ci_high[0][2]:.4g}]\n"
        f"ASD variance width (max/e): {widths[1][0]:.4g}, 99% CI: [{width_ci_low[1][0]:.4g}, {width_ci_high[1][0]:.4g}]\n"
        f"E\I variance width (max/e): {widths[1][1]:.4g}, 99% CI: [{width_ci_low[1][1]:.4g}, {width_ci_high[1][1]:.4g}]\n"
        f"NT variance width (max/e): {widths[1][2]:.4g}, 99% CI: [{width_ci_low[1][2]:.4g}, {width_ci_high[1][2]:.4g}]\n")
    savename_prefix = "logistic_func/" if si else ""
    var_fig = pl.plot_variance_over_signal_range(signal[:, 0, 0, :], nt_variance, asd_variance, ei_variance)
    pl.savefig(var_fig, savename_prefix + "variance over signal range", tight=False, ignore=var_fig.get_axes(), si=si)

    # effect of dynamic range on variance width
    effect_size_analysis(savename_prefix)


def get_variance(si, signal_noise=c.NV_SIGNAL_SD, n_signals=c.NV_NUM_S):
    noisy_signal, signal = generate_signal(signal_noise, n_signals)
    print("generating populations...")
    asd_km, nt_km = generate_km()
    print("calculating gain...")
    asd_gain, ei_gain, nt_gain = calculate_population_responses(asd_km, noisy_signal, nt_km, si)
    del nt_km, asd_km, noisy_signal
    print("calculating variance...")
    asd_variance, ei_variance, nt_variance = calculate_variance_patterns(asd_gain, ei_gain, nt_gain)
    return asd_variance, ei_variance, nt_variance, signal


def get_width(variance, s):
    threshold = (1 / np.e) * variance.max(0)
    pass_tensor = variance >= threshold
    first_pass = np.argmax(pass_tensor, axis=0)
    last_pass = np.argwhere(pass_tensor)[0][-1] + 1
    widths = s[first_pass] - s[last_pass]
    return widths


def power_analysis():
    # generate Km with variance
    asd_het = utils.get_heterogeneity_from_n(8, 16)
    nt_het = utils.get_heterogeneity_from_n(15, 16)
    het_het = 0.1
    nt_km = c.NV_KM + np.abs(np.random.normal(nt_het, het_het, size=(1, 1, 1, c.NV_PR_REPEATS))) * \
            np.random.uniform(-1, 1, size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(np.float32)
    asd_km = c.NV_KM + np.abs(np.random.normal(asd_het, het_het, size=(1, 1, 1, c.NV_PR_REPEATS))) * \
             np.random.uniform(-1, 1, size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(np.float32)
    # calculate response variance per population
    noisy_signal, signal = generate_signal()  # (c.NV_NUM_S, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)
    print("calculating gain...")
    asd_gain, _, nt_gain = calculate_population_responses(asd_km, noisy_signal, nt_km, False)
    print("calculating variance...")
    nt_variance = nt_gain.mean(1).var(1).astype(np.float32)
    del nt_gain
    asd_variance = asd_gain.mean(1).var(1).astype(np.float32)
    del asd_gain
    # calculate variance width for all
    print("calculating variance width...")
    s = np.linspace(0, 1, c.NV_NUM_S)
    asd_width = get_width(asd_variance, s)
    nt_width = get_width(nt_variance, s)
    # bootstrap and plot
    pop_sizes, power = utils.power_analysis(asd_width, nt_width, scipy.stats.mannwhitneyu, population_sizes=50)
    fig = pl.plot_power_analysis(pop_sizes, power, title="Variance width")
    pl.savefig(fig, "Variance width power analysis", tight=True, ignore=fig.get_axes(), si=True)
    pl.close_all()


def effect_size_analysis(savename_prefix):
    signal = np.repeat(np.linspace(0, 1, c.NV_NUM_S), 10 * c.NV_NUM_NEURONS * c.NV_PR_REPEATS).reshape(
        (c.NV_NUM_S, c.NV_NUM_NEURONS, 10, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal = signal + np.random.uniform(-c.NV_SIGNAL_SD, c.NV_SIGNAL_SD, size=(
        c.NV_NUM_S, c.NV_NUM_NEURONS, 10, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal[noisy_signal < 0] = 0
    noisy_signal[noisy_signal > 1] = 1
    widths = []
    widths_std = []
    n_space = np.linspace(3, 16, 25)
    for n in tqdm(n_space):
        noise_level = utils.get_heterogeneity_from_n(n, c.NV_N, c.NV_NUM_NEURONS)
        cur_width = []
        for i in range(10):
            km = c.NV_KM + noise_level * np.random.uniform(-1, 1,
                                                           size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
                np.float32)
            gain = utils.hill_func(noisy_signal, c.NV_N, km).astype(np.float32)
            variance = gain.mean(1).var(1).astype(np.float32)
            width = utils.get_width_of_var([variance.mean(1)], np.linspace(0, 1, c.NV_NUM_S))
            cur_width.append(width[0])
        widths.append(np.mean(cur_width))
        widths_std.append(np.std(cur_width))
    fig = pl.plot_variance_width_over_dynamic_range(n_space, np.array(widths), np.array(widths_std))
    pl.savefig(fig, savename_prefix + "variance width over dynamic range", tight=True, ignore=fig.get_axes(), si=True)
    pl.close_all()


def get_bootstrapped_widths(asd_variance, ei_variance, nt_variance):
    n_boot = 10000
    boot_population_choice = np.random.choice(np.arange(asd_variance.shape[1]), size=(n_boot, nt_variance.shape[-1]))
    width_boot_dist = np.zeros((n_boot, 2, 3 if ei_variance is not None else 2)).astype(
        np.float32)  # shape (bootstrap, threshold, model)
    s = np.linspace(0, 1, c.NV_NUM_S)
    print("calculating CI using bootstrap...")

    for k, pop_choice in enumerate(boot_population_choice):
        boot_asd_var = asd_variance[:, pop_choice].mean(1)
        if ei_variance is not None:
            boot_ei_var = ei_variance[:, pop_choice].mean(1)
        else:
            boot_ei_var = None
        boot_nt_var = nt_variance[:, pop_choice].mean(1)
        width_boot_dist[k] = utils.get_width_of_var([boot_asd_var, boot_ei_var, boot_nt_var], s)
    width_ci_low = np.percentile(width_boot_dist, 1, axis=0)
    width_ci_high = np.percentile(width_boot_dist, 99, axis=0)
    widths = utils.get_width_of_var([asd_variance.mean(1), ei_variance.mean(1) if ei_variance is not None else None,
                                     nt_variance.mean(1)], s)
    return width_ci_high, width_ci_low, widths


def calculate_variance_patterns(asd_gain, ei_gain, nt_gain):
    nt_variance = nt_gain.mean(1).var(1).astype(np.float32)
    del nt_gain
    asd_variance = asd_gain.mean(1).var(1).astype(np.float32)
    del asd_gain
    ei_variance = ei_gain.mean(1).var(1).astype(np.float32)
    del ei_gain
    return asd_variance, ei_variance, nt_variance


def calculate_population_responses(asd_km, noisy_signal, nt_km, si):
    if si:
        n = c.SI_NV_N
        gain_func = utils.logistic_func
    else:
        n = c.NV_N
        gain_func = utils.ei_gain_func
    nt_gain = gain_func(noisy_signal, n, nt_km).astype(np.float32)
    ei_gain = gain_func(noisy_signal, n, nt_km, c.NU).astype(np.float32)
    asd_gain = gain_func(noisy_signal, n, asd_km).astype(np.float32)
    return asd_gain, ei_gain, nt_gain


def generate_km():
    nt_km = c.NV_KM + c.NV_NT_KM_SD * np.random.uniform(-1, 1, size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
        np.float32)
    asd_km = c.NV_KM + c.NV_ASD_KM_SD * np.random.uniform(-1, 1, size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
        np.float32)
    return asd_km, nt_km


def generate_signal(signal_noise=c.NV_SIGNAL_SD, n_signals=c.NV_NUM_S):
    print("generating signal...")
    signal = np.repeat(np.linspace(0, 1, n_signals), c.NV_REPEATS * c.NV_NUM_NEURONS * c.NV_PR_REPEATS).reshape(
        (n_signals, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal = signal + np.random.normal(0, signal_noise, size=(
        n_signals, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal[noisy_signal < 0] = 0
    noisy_signal[noisy_signal > 1] = 1
    return noisy_signal, signal


def rosenberg_simulation(ax=None):
    # have increased width in ASD, no shift in peak, higher variance all-over the range, variance decreases with signal strength as
    signal = np.repeat(np.linspace(0, 1, c.NV_NUM_S), c.NV_REPEATS * c.NV_NUM_NEURONS * c.NV_PR_REPEATS).reshape(
        (c.NV_NUM_S, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal = signal + np.random.uniform(-c.NV_SIGNAL_SD, c.NV_SIGNAL_SD, size=(
        c.NV_NUM_S, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal[noisy_signal < 0] = 0
    noisy_signal[noisy_signal > 1] = 1
    nt_c = 1. + c.NV_NT_KM_SD * np.random.uniform(-1, 1, size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
        np.float32)
    asd_c = 0.75 + c.NV_NT_KM_SD * np.random.uniform(-1, 1, size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
        np.float32)
    nt_gain = utils.ei_gain_func(c.ROSENBERG_SCALING * noisy_signal, 1, 0.5, nt_c).astype(np.float32)
    asd_gain = utils.ei_gain_func(c.ROSENBERG_SCALING * noisy_signal, 1, 0.5, asd_c).astype(np.float32)
    nt_variance = nt_gain.mean(1).var(1).astype(np.float32)
    del nt_gain
    asd_variance = asd_gain.mean(1).var(1).astype(np.float32)
    del asd_gain
    width_ci_high, width_ci_low, widths = get_bootstrapped_widths(asd_variance, None, nt_variance)
    print(
        f"ASD variance width (max/2): {widths[0][0]:.4g}, 99% CI: [{width_ci_low[0][0]:.4g}, {width_ci_high[0][0]:.4g}]\n"
        f"NT variance width (max/2): {widths[0][1]:.4g}, 99% CI: [{width_ci_low[0][1]:.4g}, {width_ci_high[0][1]:.4g}]\n"
        f"ASD variance width (max/e): {widths[1][0]:.4g}, 99% CI: [{width_ci_low[1][0]:.4g}, {width_ci_high[1][0]:.4g}]\n"
        f"NT variance width (max/e): {widths[1][1]:.4g}, 99% CI: [{width_ci_low[1][1]:.4g}, {width_ci_high[1][1]:.4g}]\n")
    var_fig = pl.plot_variance_over_signal_range_rosenberg(signal[:, 0, 0, :], nt_variance, asd_variance, ax)
    if ax is None:
        pl.savefig(var_fig, "Rosenberg variance over signal range", tight=False, ignore=var_fig.get_axes(), si=True)
        pl.close_all()


if __name__ == '__main__':
    variance_simulation()
