import numpy as np
import constants as c
import utils
import plotting as pl
import scipy.stats as st
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate


def simulate_binocular_rivalry():
    print("======================================\n"
          "==== Binocular rivalry simulation ====\n"
          "======================================")
    np.random.seed(c.SEED)
    print("generating signal...")
    signal = np.zeros((c.BR_NUM_REPS, c.BR_NUM_STEPS, c.BR_N_NEURONS))
    noise = np.random.normal(scale=c.BR_SIG_SIGMA, size=(c.BR_NUM_REPS, c.BR_NUM_STEPS, 1))
    noise[:, 0] += 0.5
    signal[:, 0, :] = noise[:, 0]
    # generate random walk in [0, 1]
    print("generating random walk...")
    for i in tqdm(range(1, c.BR_NUM_STEPS)):
        signal[:, i, :] = signal[:, i - 1, :] + noise[:, i]
        signal[signal[:, i, 0] < 0, i, :] = 0
        signal[signal[:, i, 0] > 1, i, :] = 1

    print("generating neuronal population...")
    asd_km = 0.5 + np.random.normal(scale=c.BR_ASD_SIGMA_KM, size=(1, 1, c.BR_N_NEURONS))
    nt_km = 0.5 + np.random.normal(scale=c.BR_NT_SIGMA_KM, size=(1, 1, c.BR_N_NEURONS))

    # get responses
    print("calculating population responses...")
    asd_resp = utils.hill_func(signal, c.BR_N, asd_km).mean(2)
    nt_resp = utils.hill_func(signal, c.BR_N, nt_km).mean(2)

    # stats
    print("calculating statistics...")
    asd_transition_count, asd_ratios, asd_mixed_states = utils.get_binocular_rivalry_stats(asd_resp)
    nt_transition_count, nt_ratios, nt_mixed_states = utils.get_binocular_rivalry_stats(nt_resp)
    index = ["ASD", "NT"]
    ratio_df = pd.DataFrame(
        [{"mean": utils.num2print(data.mean()), "std": utils.num2print(data.std()),
          "mean_ci": utils.num2print(utils.get_ci(data, np.mean))} for
         i, data in
         enumerate([asd_ratios, nt_ratios])], index=index)
    transition_df = pd.DataFrame(
        [{"mean": utils.num2print(data.mean()), "std": utils.num2print(data.std()),
          "mean_ci": utils.num2print(utils.get_ci(data, np.mean))}
         for i, data in enumerate([asd_transition_count, nt_transition_count])], index=index)
    # print statistics
    print("Pure/Mixed state ratio:")
    print(tabulate(ratio_df, headers='keys', tablefmt='psql'))
    print("\nTransition count:")
    print(tabulate(transition_df, headers='keys', tablefmt='psql'))

    # calculate and print significance test
    ratio_res = st.wilcoxon(asd_ratios, nt_ratios, alternative="less")
    transition_res = st.wilcoxon(asd_transition_count, nt_transition_count, alternative="less")
    print(
        f"Mixed state Wilcoxon signed-rank test, statistic:{utils.num2print(ratio_res[0])}, "
        f"p-value:{utils.num2print(ratio_res[1])}\n"
        f"Transition count Wilcoxon signed-rank test, statistic:{utils.num2print(transition_res[0])}, "
        f"p-value:{utils.num2print(transition_res[1])}\n")
    # plot shift function - in the SI
    ratio_shift_fig = pl.plot_shift_func(asd_ratios, nt_ratios, "ASD vs NT Pure/mixed ratio")
    pl.savefig(ratio_shift_fig, "binocular rivalry pure-mixed ratio shift function", ignore=ratio_shift_fig.get_axes(),
               si=True, tight=True)
    transition_shift_fig = pl.plot_shift_func(asd_transition_count, nt_transition_count, "ASD vs NT transition count")
    pl.savefig(transition_shift_fig, "binocular rivalry tansition count shift function",
               ignore=transition_shift_fig.get_axes(), si=True, tight=True)

    # plot
    binocular_rivalry_fig = pl.plot_binocular_rivalry(asd_transition_count, nt_transition_count, asd_ratios, nt_ratios,
                                                      asd_resp, nt_resp, signal,5)
    pl.savefig(binocular_rivalry_fig, "binocular rivalry", shift_x=-0.05, shift_y=1.05, tight=False)


if __name__ == '__main__':
    simulate_binocular_rivalry()
