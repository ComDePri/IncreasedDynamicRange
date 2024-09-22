import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats as st
from tabulate import tabulate
from tqdm import tqdm

import constants as c
import idr_utils as utils
import plotting as pl


def effect_size_analysis():
    n_space = np.linspace(3, 16, 25)
    np.random.seed(c.SEED)
    N_REPS = 200
    signal = generate_signal()[:N_REPS]
    heterogeneities = np.zeros_like(n_space, dtype=float)
    for i, n in enumerate(tqdm(n_space, desc="Calculating heterogeneity levels")):
        heterogeneities[i] = utils.get_heterogeneity_from_n(n, c.SR_N, c.SR_N_NEURONS)
    km = 0.5 + heterogeneities[:, None, None, None] * np.random.normal(scale=1,
                                                                       size=(n_space.size, 1, 1, c.BR_N_NEURONS))
    resp = utils.hill_func(signal[None, ...], c.BR_N, km).mean(-1)
    states_mat = np.zeros_like(resp)
    states_mat[resp < c.BR_PURE_LOW_THRESH] = -1
    states_mat[c.BR_PURE_HIGH_THRESH < resp] = 1
    transitions_counts = np.zeros((n_space.size, N_REPS))
    for n in tqdm(range(n_space.size)):
        for j in range(N_REPS):
            transitions_counts[n, j] = np.count_nonzero(np.diff(states_mat[n, j][np.nonzero(states_mat[n, j])]))
    pure_states_count = np.count_nonzero(states_mat, axis=-1)
    mixed_state_count = (c.BR_NUM_STEPS - pure_states_count)
    ratio = pure_states_count / mixed_state_count
    nonzero = (mixed_state_count != 0)
    ratio[~nonzero] = np.nan
    fig, (ax1, ax2) = pl.plt.subplots(1, 2, figsize=pl.get_fig_size(1, 2))
    ax1: pl.plt.Axes
    ax1.violinplot(transitions_counts.T, positions=n_space, widths=0.5, showmeans=True)
    ax1.set_xticks(np.arange(2, 18, 2), np.arange(2, 18, 2))
    ax1.set(**{"title": "Transitions", "xlabel": "Hill coefficient", "ylabel": "# Transitions"})
    ax2.violinplot(ratio.T, positions=n_space, widths=0.5, showmeans=True)
    ax2.set_xticks(np.arange(2, 18, 2), np.arange(2, 18, 2))
    ax2.set(**{"title": "Pure to Mixed state ratio", "xlabel": "Hill coefficient",
               "ylabel": r"$\frac{t_{pure}}{t_{mixed}}$"})
    pl.savefig(fig, "Binocular rivalry effect size as a function of n", tight=True, si=True)
    pl.plt.close()
    return transitions_counts.mean(-1), transitions_counts.std(-1), ratio.mean(-1), ratio.std(-1)


def get_transition_and_ratio(resp):
    states_mat = np.zeros_like(resp)
    states_mat[resp < c.BR_PURE_LOW_THRESH] = -1
    states_mat[c.BR_PURE_HIGH_THRESH < resp] = 1
    transitions_counts = np.zeros(resp.shape[:-1])
    for i in range(resp.shape[0]):
        for j in range(resp.shape[1]):
            transitions_counts[i, j] = np.count_nonzero(np.diff(states_mat[i, j][np.nonzero(states_mat[i, j])]))
    pure_states_count = np.count_nonzero(states_mat, axis=-1)
    mixed_state_count = (c.BR_NUM_STEPS - pure_states_count)
    ratio = pure_states_count / mixed_state_count
    nonzero = (mixed_state_count != 0)
    ratio[~nonzero] = np.nan
    return transitions_counts, ratio


def get_transition_events_and_durations(resp):
    states_mat = np.zeros_like(resp)
    states_mat[resp < c.BR_PURE_LOW_THRESH] = -1
    states_mat[c.BR_PURE_HIGH_THRESH < resp] = 1
    transitions_counts = np.zeros(resp.shape[:-1])
    event_counts = np.zeros(resp.shape[:-1])
    for i in range(resp.shape[0]):
        transitions_counts[i] = np.count_nonzero(np.diff(states_mat[i][np.nonzero(states_mat[i])]))
        event_counts[i] = np.count_nonzero(np.diff(states_mat[i]))
    pure_states_count = np.count_nonzero(states_mat, axis=-1)
    mixed_state_count = (c.BR_NUM_STEPS - pure_states_count)
    return transitions_counts, event_counts, pure_states_count, mixed_state_count


def power_analysis(het_het=0.1):
    np.random.seed(c.SEED)
    N_REPS = 50
    MAX_POP = 3000
    signal = generate_signal()[:N_REPS, ..., 0]
    asd_het = utils.get_heterogeneity_from_n(8, c.SR_N, c.SR_N_NEURONS)
    nt_het = utils.get_heterogeneity_from_n(15, c.SR_N, c.SR_N_NEURONS)

    asd_hets = np.abs(asd_het + np.random.normal(0, het_het, (MAX_POP, 1, 1)))
    nt_hets = np.abs(nt_het + np.random.normal(0, het_het, (MAX_POP, 1, 1)))
    asd_ns = utils.get_effective_n_from_heterogeneity(asd_hets, 20)
    nt_ns = utils.get_effective_n_from_heterogeneity(nt_hets, 20)

    asd_resp = utils.hill_func(signal[None, ...], asd_ns, 0.5)
    nt_resp = utils.hill_func(signal[None, ...], nt_ns, 0.5)
    asd_transition_count, asd_ratios = get_transition_and_ratio(asd_resp)
    nt_transition_count, nt_ratios = get_transition_and_ratio(nt_resp)

    pop_sizes, ratio_power = utils.power_analysis(np.median(asd_ratios, axis=-1), np.median(nt_ratios, axis=-1),
                                                  scipy.stats.mannwhitneyu)
    pop_sizes, transition_power = utils.power_analysis(np.median(asd_transition_count, axis=-1),
                                                       np.median(nt_transition_count, axis=-1),
                                                       scipy.stats.mannwhitneyu)
    fig, (ax1, ax2) = pl.plt.subplots(1, 2, figsize=pl.get_fig_size(1, 2))
    pl.plot_power_analysis(pop_sizes, ratio_power, "Pure/Mixed state ratio", ax1)
    pl.plot_power_analysis(pop_sizes, transition_power, "Transition state ratio", ax2)
    pl.savefig(fig, "Binocular rivalry power analysis", tight=True, si=True)
    pl.close_all()
    return pop_sizes, ratio_power, transition_power


def fit_robertson():
    np.random.seed(c.SEED)
    signal = generate_signal()
    asd_km, ei_km, nt_km = get_populations(0.1)
    asd_resp, _, nt_resp = get_population_responses(asd_km, ei_km, nt_km, signal, False)
    asd_transition_count, asd_event_count, asd_pure, asd_mixed = get_transition_events_and_durations(asd_resp)
    nt_transition_count, nt_event_count, nt_pure, nt_mixed = get_transition_events_and_durations(nt_resp)
    nt_reversion_prop = ((nt_event_count - nt_transition_count) / nt_transition_count)
    nt_reversion_prop[nt_transition_count == 0] = np.nan
    asd_reversion_prop = ((asd_event_count - asd_transition_count) / asd_transition_count)
    asd_reversion_prop[asd_transition_count == 0] = np.nan
    pl.plt.bar(np.arange(2), [np.nanmean(nt_reversion_prop), np.nanmean(asd_reversion_prop)],
               yerr=[np.nanstd(nt_reversion_prop), np.nanstd(asd_reversion_prop)], color=[c.NT_COLOR, c.ASD_COLOR])
    pl.plt.xticks(np.arange(2), ["NDR", "IDR"])
    pl.plt.ylabel(r"Reversion proportion")
    pl.savefig(pl.plt.gcf(), "Robertson fit", tight=True, si=True)
    print(scipy.stats.mannwhitneyu(nt_reversion_prop, asd_reversion_prop, nan_policy="omit"))
    asd_sum_low = (asd_resp < c.BR_PURE_LOW_THRESH).sum(axis=-1)
    asd_sum_high = (asd_resp > c.BR_PURE_HIGH_THRESH).sum(axis=-1)
    nt_sum_low = (nt_resp < c.BR_PURE_LOW_THRESH).sum(axis=-1)
    nt_sum_high = (nt_resp > c.BR_PURE_HIGH_THRESH).sum(axis=-1)
    asd_max_time_percept = np.maximum(asd_sum_low, asd_sum_high)
    nt_max_time_percept = np.maximum(nt_sum_low, nt_sum_high)


def simulate_binocular_rivalry(si=False, plot=True):
    print("======================================\n"
          "==== Binocular rivalry simulation ====\n"
          "======================================")
    np.random.seed(c.SEED)
    signal = generate_signal()
    asd_km, ei_km, nt_km = get_populations()
    asd_resp, ei_resp, nt_resp = get_population_responses(asd_km, ei_km, nt_km, signal, si)

    # stats
    all_nonzero, asd_ratios, asd_transition_count, ei_ratios, ei_transition_count, \
        nt_ratios, nt_transition_count = transition_and_time_ratio_stats(asd_resp, ei_resp, nt_resp, si, plot)

    # plot
    save_prefix = "logistic_func/" if si else ""
    binocular_rivalry_fig = pl.plot_binocular_rivalry(asd_transition_count, nt_transition_count, ei_transition_count,
                                                      asd_ratios[all_nonzero], nt_ratios[all_nonzero],
                                                      ei_ratios[all_nonzero], asd_resp, nt_resp, ei_resp, signal, 5)
    pl.savefig(binocular_rivalry_fig, save_prefix + "binocular rivalry", shift_x=-0.05, shift_y=1.05, tight=False,
               si=si)

    # sweep effect size as a function of n

    effect_size_analysis()
    # fit_robertson()


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


def transition_and_time_ratio_stats(asd_resp, ei_resp, nt_resp, si, plot=True):
    print("calculating statistics...")
    asd_transition_count, asd_ratios, asd_mixed_states, asd_nonzero = utils.get_binocular_rivalry_stats(asd_resp)
    nt_transition_count, nt_ratios, nt_mixed_states, nt_nonzero = utils.get_binocular_rivalry_stats(nt_resp)
    if ei_resp is not None:
        ei_transition_count, ei_ratios, ei_mixed_states, ei_nonzero = utils.get_binocular_rivalry_stats(ei_resp)
        all_nonzero = asd_nonzero & nt_nonzero & ei_nonzero
        index = ["IDR", "NDR", "EI"]
    else:
        ei_ratios, ei_transition_count = None, None
        all_nonzero = asd_nonzero & nt_nonzero
        index = ["IDR", "NDR"]
    ratio_df, transition_df = get_data_frames(all_nonzero, asd_ratios, asd_transition_count, ei_ratios,
                                              ei_transition_count, index, nt_ratios, nt_transition_count)
    # print statistics
    print("Pure/Mixed state ratio:")
    print(tabulate(ratio_df, headers='keys', tablefmt='psql'))
    print("\nTransition count:")
    print(tabulate(transition_df, headers='keys', tablefmt='psql'))

    # calculate and print significance test
    if plot:
        compare_binocular_rivalry_stat_pairs(asd_ratios[all_nonzero], nt_ratios[all_nonzero], asd_transition_count,
                                             nt_transition_count, "IDR", "NDR")
        if ei_resp is not None:
            compare_binocular_rivalry_stat_pairs(asd_ratios[all_nonzero], ei_ratios[all_nonzero], asd_transition_count,
                                                 ei_transition_count, "IDR", "EI")
            compare_binocular_rivalry_stat_pairs(nt_ratios[all_nonzero], ei_ratios[all_nonzero], nt_transition_count,
                                                 ei_transition_count, "NDR", "EI")

        # plot shift function - in the SI
        binocular_rivalry_stat_pair_shift_func(asd_ratios[all_nonzero], nt_ratios[all_nonzero], asd_transition_count,
                                               nt_transition_count, "IDR", "NDR", si=si)
        if ei_resp is not None:
            binocular_rivalry_stat_pair_shift_func(asd_ratios[all_nonzero], ei_ratios[all_nonzero],
                                                   asd_transition_count,
                                                   ei_transition_count, "IDR", "EI", si=si)
            binocular_rivalry_stat_pair_shift_func(ei_ratios[all_nonzero], nt_ratios[all_nonzero], ei_transition_count,
                                                   nt_transition_count, "EI", "NDR", si=si)

    return all_nonzero, asd_ratios, asd_transition_count, ei_ratios, ei_transition_count, nt_ratios, nt_transition_count


def get_data_frames(all_nonzero, asd_ratios, asd_transition_count, ei_ratios, ei_transition_count, index, nt_ratios,
                    nt_transition_count):
    ratios = [r[all_nonzero] for r in [asd_ratios, nt_ratios, ei_ratios] if r is not None]
    transition_counts = [r[all_nonzero] for r in [asd_transition_count, nt_transition_count, ei_transition_count] if
                         r is not None]

    ratio_df = pd.DataFrame([{"mean": utils.num2print(data.mean()), "std": utils.num2print(data.std()),
                              "mean_ci": utils.num2print(utils.get_ci(data, np.mean))} for i, data in
                             enumerate(ratios)], index=index)
    transition_df = pd.DataFrame([{"mean": utils.num2print(data.mean()), "std": utils.num2print(data.std()),
                                   "mean_ci": utils.num2print(utils.get_ci(data, np.mean))} for i, data in
                                  enumerate(transition_counts)], index=index)
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


def get_populations(asd_sigma=c.BR_ASD_SIGMA_KM, nt_sigma=c.BR_NT_SIGMA_KM):
    print("generating neuronal population...")
    asd_km = 0.5 + np.random.normal(scale=asd_sigma, size=(1, 1, c.BR_N_NEURONS))
    nt_km = 0.5 + np.random.normal(scale=nt_sigma, size=(1, 1, c.BR_N_NEURONS))
    ei_km = nt_km
    return asd_km, ei_km, nt_km


def generate_signal(scale=c.BR_SIG_SIGMA):
    print("generating signal...")
    main_signal = np.zeros((c.BR_NUM_REPS, c.BR_NUM_STEPS))
    noise = np.random.normal(scale=scale, size=(c.BR_NUM_REPS, c.BR_NUM_STEPS))

    main_signal[:, 0] = noise[:, 0] + 0.5

    # generate random walk in [0, 1]
    print("generating random walk...")
    for i in tqdm(range(1, c.BR_NUM_STEPS)):
        main_signal[:, i] = main_signal[:, i - 1] + noise[:, i]
        main_signal[main_signal[:, i] < 0, i] = 0
        main_signal[main_signal[:, i] > 1, i] = 1

    return np.clip(main_signal[..., None] + np.random.normal(scale=c.BR_SIG_SIGMA,
                                                             size=(c.BR_NUM_REPS, c.BR_NUM_STEPS, c.BR_N_NEURONS)), 0,
                   1)


def rosenberg_simulation():
    # plot is ugly, but - this is reduced (log-like) mirroring of the input, the ASD have slightly more transitions and almost the same ratios.
    print("======================================\n"
          "==== Binocular rivalry simulation ====\n"
          "======================================")
    np.random.seed(c.SEED)
    signal = np.zeros((c.BR_NUM_REPS, c.BR_NUM_STEPS, c.BR_N_NEURONS))
    noise = np.random.normal(scale=c.BR_SIG_SIGMA, size=(c.BR_NUM_REPS, c.BR_NUM_STEPS, 1))
    noise[:, 0] += 0.25
    signal[:, 0, :] = noise[:, 0]
    # generate random walk in [0, 1]
    print("generating random walk...")
    for i in tqdm(range(1, c.BR_NUM_STEPS)):
        signal[:, i, :] = signal[:, i - 1, :] + noise[:, i]
        signal[signal[:, i, 0] < 0, i, :] = 0
        signal[signal[:, i, 0] > 1, i, :] = 1
    asd_c = 0.75 + np.random.normal(scale=c.BR_NT_SIGMA_KM, size=(1, 1, c.BR_N_NEURONS))
    nt_c = 1. + np.random.normal(scale=c.BR_NT_SIGMA_KM, size=(1, 1, c.BR_N_NEURONS))
    asd_resp = utils.ei_gain_func(signal, 1, 0.5, asd_c).mean(2)
    nt_resp = utils.ei_gain_func(signal, 1, 0.5, nt_c).mean(2)

    asd_transition_count, asd_ratios, asd_mixed_states, asd_nonzero = utils.get_binocular_rivalry_stats(asd_resp, 0.2,
                                                                                                        0.6)
    nt_transition_count, nt_ratios, nt_mixed_states, nt_nonzero = utils.get_binocular_rivalry_stats(nt_resp, 0.2, 0.6)

    all_nonzero = asd_nonzero & nt_nonzero
    # plot
    binocular_rivalry_fig = pl.plot_binocular_rivalry_rosenberg(asd_transition_count, nt_transition_count,
                                                                asd_ratios[all_nonzero], nt_ratios[all_nonzero],
                                                                asd_resp, nt_resp, signal, 5)
    pl.savefig(binocular_rivalry_fig, "Rosenberg binocular rivalry", shift_x=-0.05, shift_y=1.05, tight=False,
               si=True)
    pl.close_all()


# %%
if __name__ == '__main__':
    simulate_binocular_rivalry()
