from os.path import exists
import numpy as np
import constants as c
import utils
import plotting as pl
from tqdm import tqdm
import itertools as it


def simulate_separatrix():
    print("=====================================\n"
          "======= Separatrix simulation =======\n"
          "=====================================")
    # prepare parameters
    MAIN_FIG_SIGNAL_IDX = 0
    SUPP_FIG_SIGNAL_IDX = 1
    signal = np.array([0.4, 0.45])

    km_sigma_levels = np.array([0.01, 0.075, 0.175, 0.21])
    effective_n_list = utils.get_effective_n_from_heterogeneity(km_sigma_levels)
    signal_sigma_levels = np.linspace(0, .25, 41).astype(np.float64)
    time = np.arange(0, c.SEP_MAX_T, c.SEP_DT)

    param_list = [signal, signal_sigma_levels, km_sigma_levels]
    if not exists("separatrix_activated_fraction.npy"):
        # Relatively heavy computation, load if possible
        print('Generating separatrix...')
        # prepare all parameter combinations as a list
        connectivity_mat = (np.ones((c.SEP_N_NEURONS, c.SEP_N_NEURONS)) / c.SEP_N_NEURONS).astype(np.float64)
        params = list(it.product(*param_list))
        # prepare the results matrix with the correct shape
        results_mat = np.zeros([param.size for param in param_list] + [time.size, c.SEP_N_NEURONS, c.SEP_REPEATS],
                               dtype=np.float32)
        np.random.seed(c.SEED)
        for i, tup in enumerate(tqdm(params)):
            tup_idx = np.unravel_index(i, results_mat.shape[:len(param_list)])
            results_mat[tup_idx] = utils.separatrix_run(
                tup + (connectivity_mat, c.SEP_N_NEURONS, c.SEP_REPEATS))

        # 1 in each time-point the population is active
        # then 1 if the population was active at any time-point in the run
        activity_patterns = np.isclose(results_mat.mean(4), 1, atol=c.SEP_ACTIVE_TOL).max(axis=3)
        activated_fraction = activity_patterns.mean(axis=-1)  # % repeats activated
        # save results
        np.save("activity_patterns", activity_patterns)
        np.save("separatrix_activated_fraction", activated_fraction)
    else:
        print('Loading separatrix...')
        activity_patterns = np.load("activity_patterns.npy")
        activated_fraction = np.load("separatrix_activated_fraction.npy")
    # calculate bootstrap CI for the % activated and save the figure
    activated_fraction_ci = utils.get_ci(activity_patterns[MAIN_FIG_SIGNAL_IDX], np.mean, axis=-1)
    fig, ax = pl.plot_separatrix(signal_sigma_levels, activated_fraction[MAIN_FIG_SIGNAL_IDX], effective_n_list,
                                 activated_fraction_ci)
    pl.savefig(fig, "Separatrix", ignore=[ax], tight=False)
    activated_fraction_ci = utils.get_ci(activity_patterns[SUPP_FIG_SIGNAL_IDX], np.mean, axis=-1)
    fig, ax = pl.plot_separatrix(signal_sigma_levels, activated_fraction[SUPP_FIG_SIGNAL_IDX], effective_n_list,
                                 activated_fraction_ci)
    pl.savefig(fig, "Separatrix signal level 0.45", ignore=[ax], tight=False, si=True)


if __name__ == '__main__':
    simulate_separatrix()
