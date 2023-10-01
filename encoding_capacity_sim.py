import numpy as np
import constants as c
import idr_utils as utils
import plotting as pl
import scipy
import data_fitting
from scipy.io import loadmat


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


if __name__ == '__main__':
    simulate_encoding_capacity()
