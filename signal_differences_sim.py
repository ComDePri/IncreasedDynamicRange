import numpy as np
import scipy.stats

import constants as c
import idr_utils as utils
import plotting as pl


def simulate_signal_differences(si=False):
    print("======================================\n"
          "========= Signal differences ========\n"
          "======================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.SSD_NUM_S_LEVELS)
    if si:
        gain_func = utils.logistic_func
        asd_n = c.SI_SSD_ASD_N
        nt_n = c.SI_SSD_NT_N
    else:
        gain_func = utils.ei_gain_func
        asd_n = c.SSD_ASD_N
        nt_n = c.SSD_NT_N
    nt_resp = gain_func(s, nt_n, c.SSD_KM)
    asd_resp = gain_func(s, asd_n, c.SSD_KM)
    ei_resp = gain_func(s, nt_n, c.SSD_KM, c.EI_NU)
    nt_dr = [s[np.argmin(np.abs(0.1 - nt_resp))], s[np.argmin(np.abs(0.9 - nt_resp))]]
    asd_dr = [s[np.argmin(np.abs(0.1 - asd_resp))], s[np.argmin(np.abs(0.9 - asd_resp))]]
    ei_dr = [s[np.argmin(np.abs(0.1 * ei_resp.max() - ei_resp))], s[np.argmin(np.abs(0.9 * ei_resp.max() - ei_resp))]]
    print(rf"NT dynamic range: {np.round(nt_dr, 2)}, R={nt_dr[1] / nt_dr[0]:.2f}")
    print(rf"ASD dynamic range: {np.round(asd_dr, 2)}, R={asd_dr[1] / asd_dr[0]:.2f}")
    print(rf"E\I dynamic range: {np.round(ei_dr, 2)}, R={ei_dr[1] / ei_dr[0]:.2f}")
    signal_sensitivity_fig = pl.plot_sensitivity_to_signal_differences(s, nt_resp, asd_resp, None, si=si)
    savename_prefix = "logistic_func/" if si else ""
    pl.savefig(signal_sensitivity_fig, savename_prefix + "sensitivity to signal differences",
               ignore=signal_sensitivity_fig.get_axes(), tight=False, si=si)
    signal_sensitivity_fig = pl.plot_sensitivity_to_signal_differences(s, nt_resp, asd_resp, ei_resp, si=si)
    pl.savefig(signal_sensitivity_fig, savename_prefix + "sensitivity to signal differences",
               ignore=signal_sensitivity_fig.get_axes(), tight=False, si=True)
    effect_size_analysis()


def power_analysis():
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.SSD_NUM_S_LEVELS)
    het_het = 0.1
    n_neurons = 200
    n_populations = 100
    asd_het = utils.get_heterogeneity_from_n(c.SSD_ASD_N, 16, n_neurons)
    nt_het = utils.get_heterogeneity_from_n(15, 16, n_neurons)
    asd_km = (0.5 + np.abs(asd_het + np.random.normal(0, het_het, (1, n_neurons, n_populations))) *
              np.random.uniform(-1, 1, size=(1, n_neurons, n_populations)))
    nt_km = (0.5 + np.abs(nt_het + np.random.normal(0, het_het, (1, n_neurons, n_populations))) *
             np.random.uniform(-1, 1, size=(1, n_neurons, n_populations)))
    asd_resp = utils.ei_gain_func(s[:, None, None], c.SSD_ASD_N, asd_km).mean(1)
    nt_resp = utils.ei_gain_func(s[:, None, None], c.SSD_ASD_N, nt_km).mean(1)
    asd_low_differences = np.abs(asd_resp[c.SSD_LOW_SIG2_IDX, :] - asd_resp[c.SSD_LOW_SIG1_IDX, :])
    asd_high_differences = np.abs(asd_resp[c.SSD_HIGH_SIG2_IDX, :] - asd_resp[c.SSD_HIGH_SIG1_IDX, :])
    nt_low_differences = np.abs(nt_resp[c.SSD_LOW_SIG2_IDX, :] - nt_resp[c.SSD_LOW_SIG1_IDX, :])
    nt_high_differences = np.abs(nt_resp[c.SSD_HIGH_SIG2_IDX, :] - nt_resp[c.SSD_HIGH_SIG1_IDX, :])
    pop_sizes, low_diff_power = utils.power_analysis(asd_low_differences, nt_low_differences, scipy.stats.mannwhitneyu)
    pop_sizes, high_diff_power = utils.power_analysis(asd_high_differences, nt_high_differences,
                                                      scipy.stats.mannwhitneyu)
    fig, (ax1, ax2) = pl.plt.subplots(1, 2, figsize=pl.get_fig_size(1, 2))
    pl.plot_power_analysis(pop_sizes, low_diff_power, "Low signal differences", ax1)
    pl.plot_power_analysis(pop_sizes, high_diff_power, "High signal differences", ax2)
    pl.savefig(fig, "Sensitivity to signal differences power analysis", tight=False, si=True)
    pl.close_all()


def effect_size_analysis():
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.SSD_NUM_S_LEVELS)
    n_space = np.linspace(3, 16, 20)
    resps = utils.ei_gain_func(s[:, None], n_space[None, :], c.SSD_KM)
    low_differences = np.abs(resps[c.SSD_LOW_SIG2_IDX, :] - resps[c.SSD_LOW_SIG1_IDX, :])
    high_differences = np.abs(resps[c.SSD_HIGH_SIG2_IDX, :] - resps[c.SSD_HIGH_SIG1_IDX, :])
    fig, (ax1, ax2) = pl.plt.subplots(1, 2, figsize=pl.get_fig_size(1, 2))
    ax1.plot(n_space, low_differences)
    ax1.set_title("Low signal differences")
    ax1.set_xlabel("n")
    ax1.set_ylabel(r"$S_{0.3} - S_{0.2}$")
    ax2.plot(n_space, high_differences)
    ax2.set_title("High signal differences")
    ax2.set_xlabel("n")
    ax2.set_ylabel(r"$S_{0.8} - S_{0.7}$")
    pl.savefig(fig, "Sensitivity to signal differences effect size analysis", tight=False, si=True)
    pl.close_all()


def simulate_rosenberg(ax=None):
    # decreased dynamic range for ASD, yet almost the same discriminability.
    # The same when normalizing, even less of a difference.
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.SSD_NUM_S_LEVELS)
    asd_resp = utils.ei_gain_func(c.ROSENBERG_SCALING * s, 1, 0.5, 0.75)
    nt_resp = utils.ei_gain_func(c.ROSENBERG_SCALING * s, 1, 0.5, 1)
    nt_dr = [s[np.argmin(np.abs(0.1 - nt_resp))], s[np.argmin(np.abs(0.9 - nt_resp))]]
    asd_dr = [s[np.argmin(np.abs(0.1 - asd_resp))], s[np.argmin(np.abs(0.9 - asd_resp))]]
    print(f"NT dynamic range: {np.round(nt_dr, 2)}, R={nt_dr[1] / nt_dr[0]:.2f}")
    print(f"ASD dynamic range: {np.round(asd_dr, 2)}, R={asd_dr[1] / asd_dr[0]:.2f}")
    signal_sensitivity_fig = pl.plot_sensitivity_to_signal_differences_rosenberg(c.ROSENBERG_SCALING * s, nt_resp,
                                                                                 asd_resp, ax)
    if ax is None:
        pl.savefig(signal_sensitivity_fig, "Rosenberg sensitivity to signal differences",
                   ignore=signal_sensitivity_fig.get_axes(), tight=False, si=True)


if __name__ == '__main__':
    simulate_signal_differences()
