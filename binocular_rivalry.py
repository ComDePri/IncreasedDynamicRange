import numpy as np
import constants as c
import idr_utils as utils
import plotting as pl
import scipy.stats as st
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate


def simulate_binocular_rivalry(si=False):
    print("======================================\n"
          "==== Binocular rivalry simulation ====\n"
          "======================================")
    np.random.seed(c.SEED)
    signal = generate_signal()
    asd_km, ei_km, nt_km = get_populations()
    asd_resp, ei_resp, nt_resp = get_population_responses(asd_km, ei_km, nt_km, signal, si)

    # stats
    all_nonzero, asd_ratios, asd_transition_count, ei_ratios, ei_transition_count, \
        nt_ratios, nt_transition_count = transition_and_time_ratio_stats(asd_resp, ei_resp, nt_resp, si)

    # plot
    save_prefix = "logistic_func/" if si else ""
    binocular_rivalry_fig = pl.plot_binocular_rivalry(asd_transition_count, nt_transition_count, ei_transition_count,
                                                      asd_ratios[all_nonzero], nt_ratios[all_nonzero],
                                                      ei_ratios[all_nonzero], asd_resp, nt_resp, ei_resp, signal, 5)
    pl.savefig(binocular_rivalry_fig, save_prefix + "binocular rivalry", shift_x=-0.05, shift_y=1.05, tight=False,si=si)


def compare_binocular_rivalry_stat_pairs(ratio1, ratio2, transition1, transition2, name1, name2):
    ratio_res = st.wilcoxon(ratio1, ratio2, alternative="less")
    transition_res = st.wilcoxon(transition1, transition2, alternative="less")
    print(
        f"{name1}<{name2}:\nMixed state Wilcoxon signed-rank test, statistic:{utils.num2print(ratio_res[0])}, "
        f"p-value:{utils.num2print(ratio_res[1])}\n"
        f"Transition count Wilcoxon signed-rank test, statistic:{utils.num2print(transition_res[0])}, "
        f"p-value:{utils.num2print(transition_res[1])}\n")


def binocular_rivalry_stat_pair_shift_func(ratio1, ratio2, transition1, transition2, name1, name2, si=False):
    save_prefix = "logistic_func/" if si else ""
    ratio_shift_fig = pl.plot_shift_func(ratio1, ratio2, f"{name1} vs {name2} Pure/mixed ratio")
    pl.savefig(ratio_shift_fig, save_prefix + f"{name1}-{name2} binocular rivalry pure-mixed ratio shift function",
               ignore=ratio_shift_fig.get_axes(), si=True, tight=True)
    transition_shift_fig = pl.plot_shift_func(transition1, transition2, f"{name1} vs {name2} transition count")
    pl.savefig(transition_shift_fig, save_prefix + f"{name1}-{name2} binocular rivalry transition count shift function",
               ignore=transition_shift_fig.get_axes(), si=True, tight=True)


def transition_and_time_ratio_stats(asd_resp, ei_resp, nt_resp, si):
    print("calculating statistics...")
    asd_transition_count, asd_ratios, asd_mixed_states, asd_nonzero = utils.get_binocular_rivalry_stats(asd_resp)
    nt_transition_count, nt_ratios, nt_mixed_states, nt_nonzero = utils.get_binocular_rivalry_stats(nt_resp)
    ei_transition_count, ei_ratios, ei_mixed_states, ei_nonzero = utils.get_binocular_rivalry_stats(ei_resp)
    all_nonzero = asd_nonzero & nt_nonzero & ei_nonzero
    index = ["ASD", "NT", "E/I"]
    ratio_df, transition_df = get_data_frames(all_nonzero, asd_ratios, asd_transition_count, ei_ratios,
                                              ei_transition_count, index, nt_ratios, nt_transition_count)
    # print statistics
    print("Pure/Mixed state ratio:")
    print(tabulate(ratio_df, headers='keys', tablefmt='psql'))
    print("\nTransition count:")
    print(tabulate(transition_df, headers='keys', tablefmt='psql'))

    # calculate and print significance test
    compare_binocular_rivalry_stat_pairs(asd_ratios[all_nonzero], nt_ratios[all_nonzero], asd_transition_count,
                                         nt_transition_count, "ASD", "NT")
    compare_binocular_rivalry_stat_pairs(asd_ratios[all_nonzero], ei_ratios[all_nonzero], asd_transition_count,
                                         ei_transition_count, "ASD", "E/I")
    compare_binocular_rivalry_stat_pairs(nt_ratios[all_nonzero], ei_ratios[all_nonzero], nt_transition_count,
                                         ei_transition_count, "NT", "E/I")

    # plot shift function - in the SI
    binocular_rivalry_stat_pair_shift_func(asd_ratios[all_nonzero], nt_ratios[all_nonzero], asd_transition_count,
                                           nt_transition_count, "ASD", "NT", si=si)
    binocular_rivalry_stat_pair_shift_func(asd_ratios[all_nonzero], ei_ratios[all_nonzero], asd_transition_count,
                                           ei_transition_count, "ASD", "E/I", si=si)
    binocular_rivalry_stat_pair_shift_func(ei_ratios[all_nonzero], nt_ratios[all_nonzero], ei_transition_count,
                                           nt_transition_count, "E/I", "NT", si=si)
    return all_nonzero, asd_ratios, asd_transition_count, ei_ratios, ei_transition_count, nt_ratios, nt_transition_count


def get_data_frames(all_nonzero, asd_ratios, asd_transition_count, ei_ratios, ei_transition_count, index, nt_ratios,
                    nt_transition_count):
    ratio_df = pd.DataFrame(
        [{"mean": utils.num2print(data.mean()), "std": utils.num2print(data.std()),
          "mean_ci": utils.num2print(utils.get_ci(data, np.mean))} for
         i, data in
         enumerate([asd_ratios[all_nonzero], nt_ratios[all_nonzero], ei_ratios[all_nonzero]])], index=index)
    transition_df = pd.DataFrame(
        [{"mean": utils.num2print(data.mean()), "std": utils.num2print(data.std()),
          "mean_ci": utils.num2print(utils.get_ci(data, np.mean))}
         for i, data in enumerate([asd_transition_count, nt_transition_count, ei_transition_count])], index=index)
    return ratio_df, transition_df


def get_population_responses(asd_km, ei_km, nt_km, signal, si):
    print("calculating population responses...")
    if si:
        gain_func = utils.logistic_func
        n = c.SI_BR_N
    else:
        gain_func = utils.ei_gain_func
        n = c.BR_N
    asd_resp = gain_func(signal, n, asd_km).mean(2)
    nt_resp = gain_func(signal, n, nt_km).mean(2)
    ei_resp = gain_func(signal, n, ei_km, c.EI_NU).mean(2)
    return asd_resp, ei_resp, nt_resp


def get_populations():
    print("generating neuronal population...")
    asd_km = 0.5 + np.random.normal(scale=c.BR_ASD_SIGMA_KM, size=(1, 1, c.BR_N_NEURONS))
    nt_km = 0.5 + np.random.normal(scale=c.BR_NT_SIGMA_KM, size=(1, 1, c.BR_N_NEURONS))
    ei_km = nt_km
    return asd_km, ei_km, nt_km


def generate_signal():
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
    return signal


if __name__ == '__main__':
    simulate_binocular_rivalry()
