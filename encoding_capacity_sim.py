import numpy as np
import scipy
from scipy.io import loadmat

import constants as c
import data_fitting
import idr_utils as utils
import plotting as pl


def simulate_encoding_capacity(si=False):
    print("======================================\n"
          "==== Encoding capacity simulation ====\n"
          "======================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.FI_NUM_S)
    fi = []
    if si:
        n_list = c.SI_FI_N_LIST
        fi_func = utils.population_resp_fi_logistic
        nt_n = c.FI_NT_N
    else:
        n_list = c.FI_N_LIST
        fi_func = utils.population_resp_fi
        nt_n = c.FI_NT_N

    for n in n_list:
        fi.append(fi_func(s, n, c.FI_KM))
    ei_fi = []
    for nu in c.FI_NU_LIST:
        ei_fi.append(fi_func(s, nt_n, c.FI_KM, nu))

    if not si:
        # these values were calculated using the "total FI derivation.nb" mathematica notebook
        nt_n_range = [[14.4145, 15.7655], [14.6869, 16.1339], [16.4908, 18.1446]]
        asd_n_range = [[10.6789, 11.9843], [10.7668, 12.2166], [10.4222, 11.7509]]
        nt_woFB = np.array(nt_n_range[0])
        asd_woFB = np.array(asd_n_range[0])
        n_values = np.zeros((2, 5000), dtype=float)
        np.random.seed(97)
        n_values[0, :] = scipy.stats.norm(**data_fitting.get_normal_dist_from_quants(0.05, *nt_woFB)).rvs(5000)
        n_values[1, :] = scipy.stats.norm(**data_fitting.get_normal_dist_from_quants(0.05, *asd_woFB)).rvs(5000)
        nt_fi = data_fitting.total_encoding_capacity(n_values[0, :], 0.5, 1)
        asd_fi = data_fitting.total_encoding_capacity(n_values[1, :], 0.5, 1)

        angeliki_nt_total_fi = np.squeeze(loadmat('data/encoding/nt_total_fi.mat')['nt_allTotal_woFB'])
        angeliki_asd_total_fi = np.squeeze(loadmat('data/encoding/asd_total_fi.mat')['asd_allTotal_woFB'])

        fig_fi, subax1, subax2 = pl.plot_fi(fi, ei_fi, n_values[0, :], n_values[1, :], nt_fi, asd_fi,
                                            angeliki_nt_total_fi,
                                            angeliki_asd_total_fi, s)
        save_prefix = ""
    else:
        save_prefix = "logistic_func/"
        fig_fi, subax1, subax2 = pl.plot_si_fi(fi, ei_fi, s)

    pl.savefig(fig_fi, save_prefix + "encoding capacity", shift_x=-0.05, shift_y=1.05 if si else 1.2, tight=False,
               ignore=[subax1, subax2], si=si)
    pl.plt.close()

    # dynamic range sweep
    if not si:
        effect_size_analysis()
        # power_analysis()


def effect_size_analysis():
    s = np.linspace(0, 1, c.FI_NUM_S)
    n_list = np.linspace(3, 16, 25)
    stds = utils.get_n_heterogeneity(n_list)
    fi_std = utils.population_resp_fi(s[:, None, None], np.swapaxes(
        np.random.multivariate_normal(n_list, stds * np.eye(n_list.size), size=(s.size, 300)), 1, 2), 0.5)

    total_fi = scipy.integrate.simpson(y=fi_std, x=s, axis=0)
    fig, ax = pl.plt.subplots(1, 1)
    pl.shaded_errplot(n_list, total_fi.mean(-1), total_fi.std(-1), ax=ax)
    ax.set_xlabel("Hill coefficient", fontsize=14)
    ax.set_ylabel("Total encoding capacity", fontsize=14)
    ax.set_title("Total encoding capacity\nas a function of dynamic range", fontsize=16)
    pl.savefig(fig, "total encoding capacity as a function of dynamic range", tight=False, ignore=fig.get_axes(),
               si=True)
    pl.close_all()
    return total_fi.mean(-1), total_fi.std(-1)


def power_analysis(het_het=0.1):
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, 1000)
    MAX_POP = 300
    asd_n = 8
    nt_n = 15
    asd_het = utils.get_heterogeneity_from_n(asd_n, 16)
    nt_het = utils.get_heterogeneity_from_n(nt_n, 16)
    asd_rand_het = np.abs(asd_het + np.random.normal(0, het_het, MAX_POP))
    nt_rand_het = np.abs(nt_het + np.random.normal(0, het_het, MAX_POP))
    asd_ns = utils.get_effective_n_from_heterogeneity(asd_rand_het)
    nt_ns = utils.get_effective_n_from_heterogeneity(nt_rand_het)

    asd_fi = utils.population_resp_fi(s[:, None], asd_ns[None, :], 0.5)
    nt_fi = utils.population_resp_fi(s[:, None], nt_ns[None, :], 0.5)
    asd_total_fi = scipy.integrate.simpson(y=asd_fi, x=s, axis=0)
    nt_total_fi = scipy.integrate.simpson(y=nt_fi, x=s, axis=0)

    pop_sizes, power = utils.power_analysis(asd_total_fi, nt_total_fi, scipy.stats.mannwhitneyu)

    fig = pl.plot_power_analysis(pop_sizes, power, "Total encoding capacity")
    pl.savefig(fig, "total encoding capacity power analysis", tight=False, ignore=fig.get_axes(),
               si=True)
    pl.close_all()
    return pop_sizes, power


def rosenberg_simulation(ax=None, title=None):
    s = np.linspace(0.1, 1, c.FI_NUM_S)
    fi = []
    c_list = c.FI_NU_LIST
    fi_func = lambda x, k, c: (k ** 2) / (x * (((c * x) + k) ** 3))
    for c_val in c_list:
        fi.append(fi_func(s, c.FI_KM, c_val))
    non_inf = np.ones_like(s, dtype=bool)
    for fi_val in fi:
        non_inf = non_inf & (~np.isinf(fi_val))
    fi = np.asarray([fi_val[non_inf] for fi_val in fi])
    s = s[non_inf]
    if ax is None:
        fig, ax = pl.plt.subplots()
    else:
        fig = None
    mix_colors = pl.get_normed_color_mix(c.ASD_COLOR, c.NT_COLOR, c_list)
    for i, fi_val in enumerate(fi):
        ax.plot(s, fi_val, color=mix_colors[i], label=f"c={c_list[i]}")
    ax.legend()
    ax.set_xlabel("Signal level")
    ax.set_ylabel("Encoding capacity $I_{F}(S)$")
    if title is not None:
        ax.set_title(title)
    if fig is not None:
        pl.savefig(fig, "Rosenberg encoding capacity", tight=False, ignore=fig.get_axes(), si=True)
        pl.plt.close()


if __name__ == '__main__':
    simulate_encoding_capacity()
