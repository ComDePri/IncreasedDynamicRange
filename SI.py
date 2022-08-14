import numpy as np
import matplotlib.pyplot as plt
import constants as c
import plotting as pl
from scipy.optimize import curve_fit
import scipy.stats as st
import scipy.stats
import utils
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

utils.reload(c)
utils.reload(pl)
import itertools as it

# %% Logistic func simulations - main simulations with logistic function instead of Hill function
print("Other sigmoid:")


def simulate_signal_differences():
    print("======================================\n"
          "========= Signal differences ========\n"
          "======================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.SSD_NUM_S_LEVELS)
    nt_resp = utils.logistic_func(s, c.SI_SSD_NT_N, c.SI_SSD_THRESH)
    asd_resp = utils.logistic_func(s, c.SI_SSD_ASD_N, c.SI_SSD_THRESH)

    signal_sensitivity_fig = pl.plot_sensitivity_to_signal_differences(s, nt_resp, asd_resp, si=True)
    pl.savefig(signal_sensitivity_fig, "logistic_func/sensitivity to signal differences",
               ignore=signal_sensitivity_fig.get_axes(), tight=False, si=True)


def variance_simulation():
    print("======================================\n"
          "========= Variance Simulation ========\n"
          "======================================")
    np.random.seed(c.SEED)
    print("generating signal...")
    signal = np.repeat(np.linspace(0, 1, c.NV_NUM_S), c.NV_REPEATS * c.NV_NUM_NEURONS * c.NV_PR_REPEATS).reshape(
        (c.NV_NUM_S, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal = signal + np.random.uniform(-c.NV_SIGNAL_SD, c.NV_SIGNAL_SD,
                                              size=(
                                                  c.NV_NUM_S, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)).astype(
        np.float32)
    noisy_signal[noisy_signal < 0] = 0
    noisy_signal[noisy_signal > 1] = 1
    print("generating populations...")
    nt_km = c.SI_NV_THRESH + c.NV_NT_KM_SD * np.random.uniform(-1, 1,
                                                               size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
        np.float32)
    asd_km = c.SI_NV_THRESH + c.NV_ASD_KM_SD * np.random.uniform(-1, 1,
                                                                 size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
        np.float32)
    print("calculating gain...")
    nt_gain = utils.logistic_func(noisy_signal, c.SI_NV_N, nt_km).astype(np.float32)
    ei_gain = utils.ei_logistic_func(noisy_signal, c.SI_NV_N, nt_km, c.SI_NV_NU).astype(np.float32)
    asd_gain = utils.logistic_func(noisy_signal, c.SI_NV_N, asd_km).astype(np.float32)
    del nt_km, asd_km, noisy_signal
    print("calculating variance...")
    nt_variance = nt_gain.mean(1).var(1).astype(np.float32)
    del nt_gain
    asd_variance = asd_gain.mean(1).var(1).astype(np.float32)
    del asd_gain
    ei_variance = ei_gain.mean(1).var(1).astype(np.float32)
    del ei_gain
    # calculate CI of .5 and 1/e from max variance
    n_boot = 10000
    boot_population_choice = np.random.choice(np.arange(100), size=(n_boot, nt_variance.shape[-1]))
    width_boot_dist = np.zeros((n_boot, 2, 3)).astype(np.float32)  # shape (bootstrap, threshold, model)
    s = np.linspace(0, 1, c.NV_NUM_S)
    print("calculating CI using bootstrap...")
    for k, pop_choice in enumerate(boot_population_choice):
        boot_asd_var = asd_variance[:, pop_choice].mean(1)
        boot_ei_var = ei_variance[:, pop_choice].mean(1)
        boot_nt_var = nt_variance[:, pop_choice].mean(1)
        width_boot_dist[k] = utils.get_width_of_var(boot_asd_var, boot_ei_var, boot_nt_var, s)
    width_ci_low = np.percentile(width_boot_dist, 1, axis=0)
    width_ci_high = np.percentile(width_boot_dist, 99, axis=0)
    widths = utils.get_width_of_var(asd_variance.mean(1), ei_variance.mean(1), nt_variance.mean(1), s)
    print(
        f"ASD variance width (max/2): {widths[0][0]:.4g}, 99% CI: [{width_ci_low[0][0]:.4g}, {width_ci_high[0][0]:.4g}]\n"
        f"E\I variance width (max/2): {widths[0][1]:.4g}, 99% CI: [{width_ci_low[0][1]:.4g}, {width_ci_high[0][1]:.4g}]\n"
        f"NT variance width (max/2): {widths[0][2]:.4g}, 99% CI: [{width_ci_low[0][2]:.4g}, {width_ci_high[0][2]:.4g}]\n"
        f"ASD variance width (max/e): {widths[1][0]:.4g}, 99% CI: [{width_ci_low[1][0]:.4g}, {width_ci_high[1][0]:.4g}]\n"
        f"E\I variance width (max/e): {widths[1][1]:.4g}, 99% CI: [{width_ci_low[1][1]:.4g}, {width_ci_high[1][1]:.4g}]\n"
        f"NT variance width (max/e): {widths[1][2]:.4g}, 99% CI: [{width_ci_low[1][2]:.4g}, {width_ci_high[1][2]:.4g}]\n")
    var_fig = pl.plot_variance_over_signal_range(signal[:, 0, 0, :], nt_variance, asd_variance, ei_variance)
    pl.savefig(var_fig, "logistic_func/variance over signal range", si=True, tight=False, ignore=var_fig.get_axes())


def simulate_signal_change_tracking_update():
    print("=========================================\n"
          "==== Kalman filter update simulation ====\n"
          "=========================================")

    np.random.seed(c.SEED)
    print("generating signal...")
    # create noisy signal
    signal = np.zeros((c.SR_NUM_REPS, c.SR_NUM_STEPS, c.SR_N_NEURONS))
    change_timepoint = round(max(c.SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
    signal[:, :change_timepoint, :] = c.SR_SIG_MIN
    signal[:, change_timepoint:, :] = c.SR_SIG_MAX

    signal += np.random.normal(scale=c.SR_SIG_SIGMA, size=signal.shape)
    signal[signal < 0] = 0
    signal[signal > 1] = 1
    print("generating neuronal populations...")
    # create km of ASD and NT, same across repeats
    asd_km = 0.5 + c.SR_ASD_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))
    nt_km = 0.5 + c.SR_NT_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))

    dense_sig = np.linspace(0, 1, 10000)
    asd_dense_resp = np.squeeze(utils.logistic_func(dense_sig[None, :, None], c.SI_SR_N, asd_km).mean(-1))
    nt_dense_resp = np.squeeze(utils.logistic_func(dense_sig[None, :, None], c.SI_SR_N, nt_km).mean(-1))

    # get responses
    print("calculating neuronal responses...")
    asd_resp = utils.logistic_func(signal, c.SI_SR_N, asd_km)
    nt_resp = utils.logistic_func(signal, c.SI_SR_N, nt_km)
    # extract variances per step per repeat
    print("calculating neuronal response variances...")
    asd_var = asd_resp.var(-1) / c.SR_N_NEURONS
    nt_var = nt_resp.var(-1) / c.SR_N_NEURONS

    # run Kalman filter on the measured responses
    # initialize estimates and confidences:
    print("initializing Kalman filter variables...")
    asd_estimated_var = np.zeros_like(asd_var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    asd_estimated_var[:, 0] = 0.25
    nt_estimated_var = np.zeros_like(nt_var, dtype=np.float64)
    nt_estimated_var[:, 0] = 0.25
    asd_resp_estimate = np.zeros_like(asd_var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    nt_resp_estimate = np.zeros_like(nt_var, dtype=np.float64)
    asd_resp_estimate[:, 0] = asd_resp.mean(-1)[:, 0]
    nt_resp_estimate[:, 0] = nt_resp.mean(-1)[:, 0]
    asd_kg = np.zeros_like(asd_var, dtype=np.float64)
    nt_kg = np.zeros_like(nt_var, dtype=np.float64)
    # run kalman filter
    print("running Kalman filter...")
    for i in tqdm(range(1, c.SR_NUM_STEPS)):
        utils.kalman_filter_step(i, asd_kg, asd_estimated_var, asd_var, asd_resp_estimate, asd_resp.mean(-1))
        utils.kalman_filter_step(i, nt_kg, nt_estimated_var, nt_var, nt_resp_estimate, nt_resp.mean(-1))

    asd_sig_estimate = np.zeros_like(asd_resp_estimate)
    nt_sig_estimate = np.zeros_like(nt_resp_estimate)
    for i in tqdm(range(c.SR_NUM_REPS)):
        asd_sig_estimate[i] = dense_sig[np.searchsorted(asd_dense_resp[i], asd_resp_estimate[i])]
        nt_sig_estimate[i] = dense_sig[np.searchsorted(nt_dense_resp[i], nt_resp_estimate[i])]

    # plot
    kalman_fig = pl.plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate, text_x=(40, 7), text_y=(150, 300))
    pl.savefig(kalman_fig, "logistic_func/slower updating in asd", shift_x=-0.06, shift_y=1.03, tight=False,
               ignore=kalman_fig.get_axes(), si=True)
    # statistics - time to 0.5/0.9 of the signal change
    asd_time_to_09_sig = np.argmin(np.abs(asd_sig_estimate[change_timepoint:] - (c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX)),
                                   1) - change_timepoint
    nt_time_to_09_sig = np.argmin(np.abs(nt_sig_estimate[change_timepoint:] - (c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX)),
                                  1) - change_timepoint
    print(f"ASD RT, mean: {asd_time_to_09_sig.mean()}, STD: {asd_time_to_09_sig.std()}\n",
          f"NT RT, mean: {nt_time_to_09_sig.mean()}, STD: {nt_time_to_09_sig.std()}\n")
    print(
        f"Wilcoxon signed-rank test for ASD and NT time to 0.9 of signal after change: {scipy.stats.wilcoxon(asd_time_to_09_sig, nt_time_to_09_sig)}")
    print("Last Km normality test: ")

    # Check normality assumption to use correct statistics.
    _, nt_normal_p = scipy.stats.shapiro(asd_time_to_09_sig)
    _, asd_normal_p = scipy.stats.shapiro(nt_time_to_09_sig)
    print(f"ASD normality: {asd_normal_p}, NT normality: {nt_normal_p}")
    if nt_normal_p < 0.05 or asd_normal_p < 0.05:
        print("Can't calculate F-test as normality assumption fails!\n"
              "Calculating Shift-function & permutation test for variance instead...")
        shift_fig = pl.plot_shift_func(asd_time_to_09_sig, nt_time_to_09_sig,
                                       "Reaction time to abrupt change, ASD to NT")
        pl.savefig(shift_fig, save_name="logistic ASD to NT shift function of reaction time to abrupt change",
                   ignore=shift_fig.get_axes(), si=True)
        p, permutation_fig = utils.permutation_test(asd_time_to_09_sig, nt_time_to_09_sig,
                                                    lambda x, y: x.var() - y.var(), alternative="greater", plot=True,
                                                    center_dists=True, hist_color=c.EI_COLOR, line_color=c.ASD_COLOR)
        print(
            f"Reaction time variance, ASD > NT hypothesis permutation test, diff={asd_time_to_09_sig.var() - nt_time_to_09_sig.var()}, p = {utils.num2print(p)}")
        pl.savefig(permutation_fig, "logistic Reaction time variance permutation test",
                   ignore=permutation_fig.get_axes(), si=True)
    else:
        var_stat, var_p_val = utils.f_test(asd_time_to_09_sig, nt_time_to_09_sig, "greater")
        print(f"Variance equality F-test, F: {var_stat:.4g}, p-value: {var_p_val:.4g}")


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
    asd_resp = utils.logistic_func(signal, c.SI_BR_N, asd_km).mean(2)
    nt_resp = utils.logistic_func(signal, c.SI_BR_N, nt_km).mean(2)

    # stats
    print("calculating statistics...")
    asd_transition_count, asd_ratios, asd_mixed_states = utils.get_binocular_rivalry_stats(asd_resp)
    nt_transition_count, nt_ratios, nt_mixed_states = utils.get_binocular_rivalry_stats(nt_resp)
    index = ["ASD", "NT"]
    ratio_df = pd.DataFrame(
        [{"mean": data.mean(), "std": data.std(),
          "mean_ci": utils.get_ci(data, np.mean)} for
         i, data in
         enumerate([asd_ratios, nt_ratios])], index=index)
    transition_df = pd.DataFrame(
        [{"mean": data.mean(), "std": data.std(),
          "mean_ci": utils.get_ci(data, np.mean)}
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
    # plot
    binocular_rivalry_fig = pl.plot_binocular_rivalry(asd_transition_count, nt_transition_count, asd_ratios, nt_ratios,
                                                      asd_resp, nt_resp, signal, 5)
    pl.savefig(binocular_rivalry_fig, "logistic_func/binocular rivalry", shift_x=-.125, shift_y=1.07, tight=False,
               si=True)


def simulate_hebbian_learning():
    print("=====================================\n"
          "==== Hebbian learning simulation ====\n"
          "=====================================")

    np.random.seed(c.SEED)
    time = np.arange(0, c.LR_MAX_T + 1, c.LR_DT)
    # parallel run
    Km = []
    for _ in tqdm(range(c.LR_N_TRIALS), desc="Trials"):
        Km.append(utils.run_learning_trial_logistic())
    Km = np.array(Km)
    NT_IDX = 0
    ASD_IDX = 1

    time_to_90 = np.argmax(Km >= c.LR_THRESHOLD * c.LR_THRESHOLD_PERCENTAGE, axis=1)
    sharp_lr = time_to_90[..., NT_IDX]
    gradual_lr = time_to_90[..., ASD_IDX]

    wilcoxon_res = scipy.stats.wilcoxon(sharp_lr, gradual_lr, alternative="greater")
    print(f"ASD learning rate, {gradual_lr.mean():.5g} \pm Std.={gradual_lr.std():.5g}\n"
          f"NT learning rate, {sharp_lr.mean():.5g} \pm Std.={sharp_lr.std():.5g}\n"
          f"Wilcoxon signed-rank test, W({sharp_lr.size - 1})={wilcoxon_res[0]:.5g}, ${utils.num2latex(wilcoxon_res[1])}$")
    lr_fig, subax = pl.plot_learning_rate_and_accuracy(Km, time, None, asd_hist_text=(0.485, 7),
                                                       nt_hist_text=(0.495, 20))
    pl.savefig(lr_fig, "logistic_func/learning rate and accuracy", ignore=[subax], shift_x=-0.1, shift_y=1.05,
               tight=False, si=True)

    last_km_nt = Km[:, -1, NT_IDX]
    last_km_asd = Km[:, -1, ASD_IDX]
    print(f"ASD bias, {0.5 - last_km_asd.mean():.5g} \pm Std.={last_km_asd.std():.5g}\n"
          f"NT bias, {0.5 - last_km_nt.mean():.5g}$ \pm Std. ={last_km_nt.std():.5g}")

    print("Last Km normality test: ")
    nt_normal_stat, nt_normal_p = scipy.stats.shapiro(last_km_nt)
    asd_normal_stat, asd_normal_p = scipy.stats.shapiro(last_km_asd)
    print(f"ASD normality, W={asd_normal_stat}, p<{utils.num2latex(asd_normal_p)}, "
          f"NT normality, W={nt_normal_stat}, p<{utils.num2latex(nt_normal_p)},")
    if nt_normal_p < 0.05 or asd_normal_p < 0.05:
        print("last-learned threshold, Wilcoxon signed-rank test:",
              scipy.stats.wilcoxon(0.5 - last_km_nt, 0.5 - last_km_asd, alternative="less"))
        print("Can't calculate F-test as normality assumption fails!")
    else:
        print("last-learned threshold, paired t-test:",
              scipy.stats.ttest_rel(0.5 - last_km_nt, 0.5 - last_km_asd, alternative="less"))
        var_stat, var_p_val = utils.f_test(last_km_asd, last_km_nt, "greater")
        print(f"Variance equality F-test, ASD var: {last_km_asd.var():.4g}, NT var: {last_km_nt.var():.4g}\n"
              f"F({last_km_asd.size - 1},{last_km_nt.size - 1})={var_stat:.4g}, "
              f"p<{utils.num2latex(var_p_val)}")


def simulate_encoding_capacity():
    print("======================================\n"
          "==== Encoding capacity simulation ====\n"
          "======================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.FI_NUM_S)
    fi = []
    for n in c.SI_FI_N_LIST:
        fi.append(utils.population_resp_fi_logistic(s, n, c.FI_KM))
    fig_fi = pl.plot_fi(fi, s, si=True)
    pl.savefig(fig_fi, "logistic_func/encoding capacity", shift_x=-0.08, shift_y=1.05, tight=False,
               ignore=fig_fi.get_axes(), si=True)


def simulate_population_response():
    print("========================================\n"
          "==== Population response simulation ====\n"
          "========================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.PR_NUM_S_LEVELS)
    population_s = np.repeat(s, c.PR_N_NEURONS).reshape((c.PR_N_NEURONS,) + s.shape, order='F')
    print("Calculating neurons gain functions...")
    nt_resp = utils.logistic_func(population_s,
                                  c.SI_PR_SLOPE,
                                  c.SI_PR_THRESH + c.PR_NT_K_STD * np.random.uniform(-1, 1,
                                                                                     size=(c.PR_N_NEURONS, 1)))
    asd_resp = utils.logistic_func(population_s,
                                   c.SI_PR_SLOPE,
                                   c.SI_PR_THRESH + c.PR_ASD_K_STD * np.random.uniform(-1, 1, size=(c.PR_N_NEURONS, 1)))

    fig_population_resp = pl.plot_population_response(s, asd_resp, nt_resp)
    pl.savefig(fig_population_resp, "logistic_func/NT vs ASD population response", shift_x=0, shift_y=1.01, tight=False,
               si=True)

    effective_n_asd = utils.get_effective_n(asd_resp.mean(0), np.squeeze(s), c.PR_KM)
    effective_n_nt = utils.get_effective_n(nt_resp.mean(0), np.squeeze(s), c.PR_KM)
    print("ASD effective n=%.2f, NT effective n =%.2f" % (effective_n_asd, effective_n_nt))
    asd_min_sig = s[np.argmax(asd_resp.mean(0) >= 0.1)]
    asd_max_sig = s[np.argmax(asd_resp.mean(0) >= 0.9)]
    nt_min_sig = s[np.argmax(nt_resp.mean(0) > 0.1)]
    nt_max_sig = s[np.argmax(nt_resp.mean(0) > 0.9)]
    print("ASD dynamic range: [%.4f,%.4f]\nNT dynamic range: [%.4f,%.4f],R_ASD: %.2f, R_NT:%.2f" % (
        asd_min_sig, asd_max_sig, nt_min_sig, nt_max_sig, asd_max_sig / asd_min_sig, nt_max_sig / nt_min_sig))


def simulate_separatrix():
    print("=====================================\n"
          "======= Separatrix simulation =======\n"
          "=====================================")
    np.random.seed(c.SEED)

    MAIN_FIG_SIGNAL_IDX = 0
    SUPP_FIG_SIGNAL_IDX = 1
    signal = np.array([0.4, 0.45])

    km_sigma_levels = np.array([0.01, 0.075, 0.175, 0.21])
    effective_n_list = utils.get_effective_log_n_from_heterogeneity(km_sigma_levels)
    signal_sigma_levels = np.linspace(0, .25, 41).astype(np.float64)
    time = np.arange(0, c.SEP_MAX_T, c.SEP_DT)

    param_list = [signal, signal_sigma_levels, km_sigma_levels]
    print('Generating separatrix...')
    connectivity_mat = (np.ones((c.SEP_N_NEURONS, c.SEP_N_NEURONS)) / c.SEP_N_NEURONS).astype(np.float64)
    params = list(it.product(*param_list))
    results_mat = np.zeros([param.size for param in param_list] + [time.size, c.SEP_N_NEURONS, c.SEP_REPEATS],
                           dtype=np.float32)
    for i, tup in enumerate(tqdm(params)):
        tup_idx = np.unravel_index(i, results_mat.shape[:len(param_list)])
        results_mat[tup_idx] = utils.separatrics_run_logistic(
            tup + (connectivity_mat, c.SEP_N_NEURONS, c.SEP_REPEATS))

    # 1 in each timepoint the population is active, then 1 if the population was active at any time in the run
    activity_patterns = np.isclose(results_mat.mean(4), 1, atol=c.SEP_ACTIVE_TOL).max(axis=3)

    activated_fraction = activity_patterns.mean(axis=-1)  # % repeats activated
    activated_fraction_ci = utils.get_ci(activity_patterns[MAIN_FIG_SIGNAL_IDX], np.mean, axis=-1)
    fig, ax = pl.plot_separatrix(signal_sigma_levels, activated_fraction[MAIN_FIG_SIGNAL_IDX], effective_n_list,
                                 activated_fraction_ci)
    pl.savefig(fig, "logistic_func/Separatrix", ignore=[ax], tight=False, si=True)
    activated_fraction_ci = utils.get_ci(activity_patterns[SUPP_FIG_SIGNAL_IDX], np.mean, axis=-1)
    fig, ax = pl.plot_separatrix(signal_sigma_levels, activated_fraction[SUPP_FIG_SIGNAL_IDX], effective_n_list,
                                 activated_fraction_ci)
    pl.savefig(fig, "logistic_func/Separatrix signal level 0.45", ignore=[ax], tight=False, si=True)


utils.reload(c)
utils.reload(pl)

simulate_signal_differences()  # Sensitivity to signal differences

variance_simulation()  # Variance over signal range - E/I VS IDR

simulate_signal_change_tracking_update()  # Slower responses to sharp transitions using kalman filter

simulate_binocular_rivalry()  # binocluar rivalry - noisy signal around 0.5

simulate_hebbian_learning()  # learning rate and accuracy

simulate_encoding_capacity()  # FI based encoding capacity

simulate_population_response()  # Population response

simulate_separatrix()  # separatrix

# %% check the width/R as a function of slope
signal = np.linspace(0, 1, c.PR_NUM_S_LEVELS)
slopes = np.linspace(3.2, 20, 50)
slope_resp = utils.hill_func(signal[:, None], slopes[None, :], c.PR_KM)
asd_min_sig = signal[np.argmax(slope_resp >= 0.1, axis=0)]
asd_max_sig = signal[np.argmax(slope_resp >= 0.9, axis=0)]
widths = asd_max_sig - asd_min_sig
ratios = asd_max_sig / asd_min_sig
slopes_fig = pl.plot_dynamic_range_as_function_of_slope(slopes, widths, ratios)
pl.savefig(slopes_fig, "Dynamic range as a function of slope", ignore=slopes_fig.get_axes(), si=True, tight=True)

# %% neural variability - test the effect of different levels of signal noise on variance pattern
signal_sd_list = np.round([0.07, 0.19, 0.32], 4)  # levels of signal noise
np.random.seed(c.SEED)
print("generating signal...")
signal = np.repeat(np.linspace(0, 1, c.NV_NUM_S),
                   (c.NV_REPEATS // 2) * c.NV_NUM_NEURONS * (c.NV_PR_REPEATS // 2)).reshape(
    (c.NV_NUM_S, c.NV_NUM_NEURONS, c.NV_REPEATS // 2, c.NV_PR_REPEATS // 2)).astype(np.float32)
print("generating populations...")
nt_km = c.NV_KM + c.NV_NT_KM_SD * np.random.uniform(-1, 1,
                                                    size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS // 2)).astype(
    np.float32)
asd_km = c.NV_KM + c.NV_ASD_KM_SD * np.random.uniform(-1, 1,
                                                      size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS // 2)).astype(
    np.float32)
fig, axes = pl.get_3_axes_with_3rd_centered()
max_var = 0
sig = signal[:, 0, 0, :]
# variance simulation and plot with different levels of signal noise
for i, signal_sd in enumerate(tqdm(signal_sd_list, desc="Signal SD")):
    np.random.seed(c.SEED)
    noisy_signal = signal + np.random.uniform(-signal_sd, signal_sd,
                                              size=(c.NV_NUM_S,
                                                    c.NV_NUM_NEURONS,
                                                    c.NV_REPEATS // 2,
                                                    c.NV_PR_REPEATS // 2)).astype(np.float32)
    noisy_signal[noisy_signal < 0] = 0
    noisy_signal[noisy_signal > 1] = 1
    nt_gain = utils.hill_func(noisy_signal, c.NV_N, nt_km).astype(np.float32)
    ei_gain = utils.ei_gain_func(noisy_signal, c.NV_N, nt_km, c.NU).astype(np.float32)
    asd_gain = utils.hill_func(noisy_signal, c.NV_N, asd_km).astype(np.float32)
    del noisy_signal
    nt_variance = nt_gain.mean(1).var(1).astype(np.float32)
    del nt_gain
    asd_variance = asd_gain.mean(1).var(1).astype(np.float32)
    del asd_gain
    ei_variance = ei_gain.mean(1).var(1).astype(np.float32)
    del ei_gain
    ax = axes[i]
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    if i > 5:
        ax.set_xlabel("Signal level", fontsize=27)
    if i % 3 == 0:
        ax.set_ylabel("Variance in population responses", fontsize=27)

    ax.set_title(r"$\sigma^2_{signal}=%.2f$" % signal_sd)
    scatt_nt = ax.scatter(sig, nt_variance, s=1, alpha=0.4, c=c.NT_COLOR)
    ax.plot(sig[:, 0], nt_variance.mean(1), alpha=0.8, c=c.NT_COLOR, label="NT")
    scatt_asd = ax.scatter(sig, asd_variance, s=1, alpha=0.4, c=c.ASD_COLOR)
    ax.plot(sig[:, 0], asd_variance.mean(1), alpha=0.8, c=c.ASD_COLOR, label="ASD")
    scatt_ei = ax.scatter(sig, ei_variance, s=1, alpha=0.4, c=c.EI_COLOR)
    ax.plot(sig[:, 0], ei_variance.mean(1), alpha=0.8, c=c.EI_COLOR, label="E/I")
    lgnd = ax.legend([scatt_nt, scatt_asd, scatt_ei], ["NT", "ASD", "E/I"], fontsize=30)
    for handle in lgnd.legendHandles:
        handle.set_sizes([200.0])
    max_var = max(max_var, nt_variance.max(), asd_variance.max(), ei_variance.max())
for ax in axes:
    ax.set_ylim(0, max_var * 1.1)
pl.savefig(fig, "Variance pattern with different levels of signal noise", tight=True, si=True,
           ignore=np.array(fig.get_axes()).ravel())

# %% Bayesian tracking - effect of slope on time to change
signal = np.zeros((c.SR_NUM_REPS, c.SR_NUM_STEPS, c.SR_N_NEURONS))
change_timepoint = round(max(c.SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
signal[:, :change_timepoint, :] = c.SR_SIG_MIN
signal[:, change_timepoint:, :] = c.SR_SIG_MAX

signal += np.random.normal(scale=c.SR_SIG_SIGMA, size=signal.shape)
signal[signal < 0] = 0
signal[signal > 1] = 1

noise_levels = np.linspace(0.01, 0.21, 20)  # correspond to N of 2.5 to 100% of N
time_to_09_sig = np.zeros_like(noise_levels).astype(np.float32)
effective_ns = np.zeros_like(noise_levels).astype(np.float32)
n_reps = 1
for i, noise in enumerate(tqdm(noise_levels, desc="Noise levels")):
    for k in range(n_reps):
        np.random.seed(c.SEED)
        # 0.2-0.8 instead of 0-1 so the fit is around the Km, which is the important part for the slope
        effective_n_sig = np.linspace(0.2, 0.8, 150)
        km = 0.5 + noise * np.random.uniform(-1, 1, size=(1, 1, c.SR_N_NEURONS))
        effective_n_population_resp = utils.hill_func(effective_n_sig[:, None], c.SR_N, np.squeeze(km)[None, :])
        effective_ns[i] += utils.get_effective_n(effective_n_population_resp.mean(1), effective_n_sig, 0.5)
        resp = utils.hill_func(signal, c.SR_N, km)
        pop_resp = resp.mean(-1)
        var = resp.var(-1) / c.SR_N_NEURONS
        estimated_var = np.zeros_like(var)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
        estimated_var[:, 0] = 0.25
        sig_estimate = np.zeros_like(var)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
        sig_estimate[:, 0] = pop_resp[:, 0]
        kg = np.zeros_like(var)
        for j in range(1, c.SR_NUM_STEPS):
            utils.kalman_filter_step(j, kg, estimated_var, var, sig_estimate, pop_resp)
        tmp = (np.argmin(np.abs(sig_estimate[change_timepoint:] - (c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX)),
                         1) - change_timepoint) + 1
        time_to_09_sig[i] += np.mean(tmp[tmp > 0])
    effective_ns[i] /= n_reps
    time_to_09_sig[i] /= n_reps
time_to_track_fig = pl.plot_time_to_track_as_a_function_of_n(time_to_09_sig, effective_ns)
pl.savefig(time_to_track_fig, "Time to track signal change as a function of N", ignore=time_to_track_fig.get_axes(),
           si=True)


# %% Slower updating - effect of change probability

def kalman_filter_step(t, kg, est_var, meas_var, est_sig, sig, prior_variance=c.SR_PRIOR_VARIANCE,
                       perturb_prob=c.SR_PERTURBATION_PROB):
    """
    Performs an in-place calculation of Kalman filter estimation
    :param t: Timestep to calculate the Kalman filter step for
    :param kg: Kalman gain, array of shape (..., time)
    :param est_var: estimated variance, array of shape (..., time)
    :param meas_var: measured variance, array of shape (..., time)
    :param est_sig: estimated signal, array of shape (..., time)
    :param sig: measured variance, array of shape (..., time,N), where N is number of measurements getting the same signal
    """
    delta = perturb_prob * prior_variance
    last_est_var = (1 - perturb_prob) * est_var[..., t - 1]
    kg[:, t] = (last_est_var + delta) / (
            (last_est_var + delta) + meas_var[..., t])
    est_var[:, t] = (1 - kg[..., t]) * (last_est_var + delta)
    est_sig[:, t] = est_sig[..., t - 1] + kg[..., t] * (
            sig[..., t] - est_sig[..., t - 1])


def si_plot_kalman_filter(axs, signal, asd_sig_estimate, nt_sig_estimate, text_x, text_y) -> plt.Figure:
    change_timepoint = round(max(c.SI_SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
    asd_times = np.argmax(asd_sig_estimate >= c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX, axis=-1) - change_timepoint + 1
    asd_times[asd_times < 0] = c.SI_SR_NUM_STEPS
    nt_times = np.argmax(nt_sig_estimate >= c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX, axis=-1) - change_timepoint + 1
    nt_times[nt_times < 0] = c.SI_SR_NUM_STEPS

    def get_bins(array, max_bins):
        if np.unique(array).size < max_bins:
            return np.unique(array).size
        else:
            return np.linspace(array.min() * 0.9, array.max() * 1.1, 100)

    axs.plot(asd_sig_estimate[c.SR_PLOT_REP_IDX, :], label="ASD", color=c.ASD_COLOR, linewidth=3)
    axs.plot(nt_sig_estimate[c.SR_PLOT_REP_IDX, :], label="NT", color=c.NT_COLOR, linewidth=2, alpha=0.7)
    axs.plot(signal[c.SR_PLOT_REP_IDX, :, 0], color='gray', label="measured signal", alpha=0.3)
    axs.plot([c.SR_SIG_MIN] * change_timepoint +
             [c.SR_SIG_MAX] * (c.SI_SR_NUM_STEPS - change_timepoint), color='k',
             label="real signal", linewidth=3, zorder=1)

    subax = pl.inset_axes(axs, "50%", "50%", loc="lower right", borderpad=3, bbox_transform=axs.transAxes,
                          bbox_to_anchor=(0.05, 0.05, 1, 1))  # type: plt.Axes
    subax.hist(asd_times, bins=get_bins(asd_times, 50), label="ASD", density=True)
    subax.hist(nt_times, bins=get_bins(nt_times, 50), label="NT", density=True)
    subax.text(text_x[0], text_y[0], "ASD", color=c.ASD_COLOR, size=25, weight="bold")
    subax.text(text_x[1], text_y[1], "NT", color=c.NT_COLOR, size=25, weight="bold")

    subax.set(xlabel="Time after switch (steps)", ylabel="Frequency")
    subax.xaxis.set_label_text("Time after switch (steps)", fontsize=20)
    subax.yaxis.set_label_text("Density", fontsize=20)

    return subax


def si_signal_tracking(perturb_prob=c.SR_PERTURBATION_PROB, sigma_new=c.SR_PRIOR_VARIANCE):
    print("=========================================\n"
          "==== Kalman filter update simulation ====\n"
          "=========================================")

    np.random.seed(c.SEED)
    print("generating signal...")
    # create noisy signal
    signal = np.zeros((c.SR_NUM_REPS, c.SI_SR_NUM_STEPS, c.SR_N_NEURONS))
    change_timepoint = round(max(c.SI_SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
    signal[:, :change_timepoint, :] = c.SR_SIG_MIN
    signal[:, change_timepoint:, :] = c.SR_SIG_MAX

    signal += np.random.normal(scale=c.SR_SIG_SIGMA, size=signal.shape)
    signal[signal < 0] = 0
    signal[signal > 1] = 1
    print("generating neuronal populations...")
    # create km of ASD and NT, same across repeats
    asd_km = 0.5 + c.SR_ASD_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))
    nt_km = 0.5 + c.SR_NT_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))

    dense_sig = np.linspace(0, 1, 10000)
    asd_dense_resp = np.squeeze(utils.hill_func(dense_sig[None, :, None], c.SR_N, asd_km).mean(-1))
    nt_dense_resp = np.squeeze(utils.hill_func(dense_sig[None, :, None], c.SR_N, nt_km).mean(-1))

    # get responses
    print("calculating neuronal responses...")
    asd_resp = utils.hill_func(signal, c.SR_N, asd_km)
    nt_resp = utils.hill_func(signal, c.SR_N, nt_km)
    # extract variances per step per repeat
    print("calculating neuronal response variances...")
    asd_var = asd_resp.var(-1) / c.SR_N_NEURONS
    nt_var = nt_resp.var(-1) / c.SR_N_NEURONS

    # run Kalman filter on the measured responses
    # initialize estimates and confidences:
    print("initializing Kalman filter variables...")
    asd_estimated_var = np.zeros_like(asd_var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    asd_estimated_var[:, 0] = sigma_new
    nt_estimated_var = np.zeros_like(nt_var, dtype=np.float64)
    nt_estimated_var[:, 0] = sigma_new
    asd_resp_estimate = np.zeros_like(asd_var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    nt_resp_estimate = np.zeros_like(nt_var, dtype=np.float64)
    asd_resp_estimate[:, 0] = asd_resp.mean(-1)[:, 0]
    nt_resp_estimate[:, 0] = nt_resp.mean(-1)[:, 0]
    asd_kg = np.zeros_like(asd_var, dtype=np.float64)
    nt_kg = np.zeros_like(nt_var, dtype=np.float64)
    # run kalman filter
    print("running Kalman filter...")
    for i in tqdm(range(1, c.SI_SR_NUM_STEPS)):
        utils.kalman_filter_step(i, asd_kg, asd_estimated_var, asd_var, asd_resp_estimate, asd_resp.mean(-1),
                                 perturb_prob=perturb_prob, prior_variance=sigma_new)
        utils.kalman_filter_step(i, nt_kg, nt_estimated_var, nt_var, nt_resp_estimate, nt_resp.mean(-1),
                                 perturb_prob=perturb_prob, prior_variance=sigma_new)

    asd_sig_estimate = np.zeros_like(asd_resp_estimate)
    nt_sig_estimate = np.zeros_like(nt_resp_estimate)
    for i in tqdm(range(c.SR_NUM_REPS)):
        asd_sig_estimate[i] = dense_sig[np.searchsorted(asd_dense_resp[i], asd_resp_estimate[i])]
        nt_sig_estimate[i] = dense_sig[np.searchsorted(nt_dense_resp[i], nt_resp_estimate[i])]

    return signal, asd_sig_estimate, nt_sig_estimate


perturb_prob_list = np.logspace(-5, -7, 4)

signals, asd_sig_estimates, nt_sig_estimates = [], [], []
for i, cutoff in enumerate(perturb_prob_list):
    signal, asd_sig_estimate, nt_sig_estimate = si_signal_tracking(cutoff)
    signals.append(signal)
    asd_sig_estimates.append(asd_sig_estimate)
    nt_sig_estimates.append(nt_sig_estimate)
# %%
text_x_list = [
    [15, 3.3], [35, 3.3],
    [75, 7], [190, 15]
]
# ASD,NT
text_y_list = [
    [0.4, 0.75], [0.25, 0.75],
    [0.2, 0.6], [0.2, 0.6]
]
kalman_fig = plt.figure(figsize=(pl.get_fig_size(2.5, 2.5)))
axes = kalman_fig.subplots(2, 2, sharex="all", sharey="all").ravel()
ignore_axes = []
for i, cutoff in enumerate(perturb_prob_list):
    ignore_axes.append(
        si_plot_kalman_filter(axes[i], signals[i], asd_sig_estimates[i], nt_sig_estimates[i], text_x_list[i],
                              text_y_list[i]))
    axes[i].set_title(f"Perturbation probability={utils.num2latex(cutoff)}")
    axes[i].tick_params('both', length=20, width=2, which='major')
    axes[i].tick_params('both', length=15, width=2, which='minor')
    # if i % 3 == 0:
    #     axes[i].set_ylabel("Signal")
    # if i > 5:
    #     axes[i].set_xlabel("Timepoints")
pl.savefig(kalman_fig, "Effect of perturbation probability and prior variance on tracking", tight=False,
           ignore=ignore_axes, si=True)
plt.close(kalman_fig)
# %% calculate LR, bias, bias std as a function of N:
print("\nCalculating LR, bias and bias Std. as a function of slope...\n")
time = np.arange(0, c.LR_MAX_T + 1, c.LR_DT)
n_range = np.linspace(6, 16, 20)
n_range_km = []
np.random.seed(c.SEED)
for _ in tqdm(range(c.LR_N_TRIALS), desc="Trials"):
    n_range_km.append(utils.run_learning_trial(n=n_range))
n_range_km = np.array(n_range_km)
function_of_n_fig = pl.plot_learning_as_function_of_n(n_range, n_range_km, time)
pl.savefig(function_of_n_fig, "LR, bias and bias std as function of n", si=True,
           ignore=function_of_n_fig.get_axes(), tight=True)
# %% Binocular rivalry - changing threshold levels
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
asd_resp = utils.hill_func(signal, c.BR_N, asd_km).mean(2)
nt_resp = utils.hill_func(signal, c.BR_N, nt_km).mean(2)

low_thresholds = np.linspace(0.1, 0.4, 7)
high_thresholds = 1 - low_thresholds
asd_transition_counts = np.zeros((signal.shape[0], low_thresholds.size), dtype=np.int32)
nt_transition_counts = asd_transition_counts.copy()
asd_ratios = np.zeros((signal.shape[0], low_thresholds.size), dtype=np.float64)
nt_ratios = asd_ratios.copy()
# calculate num of transitions and pure state ratio for differing levels of threshold
for i in range(low_thresholds.size):
    asd_transition_counts[:, i], asd_ratios[:, i], _ = utils.get_binocular_rivalry_stats(asd_resp,
                                                                                         low_thresh=low_thresholds[i],
                                                                                         high_thresh=high_thresholds[i])
    nt_transition_counts[:, i], nt_ratios[:, i], _ = utils.get_binocular_rivalry_stats(nt_resp,
                                                                                       low_thresh=low_thresholds[i],
                                                                                       high_thresh=high_thresholds[i])
asd_nan_rows, asd_nan_cols = np.where(np.isinf(asd_ratios))
nt_nan_rows, nt_nan_cols = np.where(np.isinf(nt_ratios))
nan_rows = np.unique(np.concatenate([asd_nan_rows, nt_nan_rows]))
if nan_rows.size > 0:
    nan_idx_vec = np.ones(asd_ratios.shape[0], dtype=bool)
    nan_idx_vec[nan_rows] = False
    asd_ratios = asd_ratios[nan_idx_vec, :]
    nt_ratios = nt_ratios[nan_idx_vec, :]

pl.plot_boxplots(boxplot_value_list=[nt_ratios, asd_ratios],
                 x_labels=np.round(low_thresholds, 2),
                 group_labels=["NT", "ASD"],
                 title="pure-mixed ratio over threshold",
                 xlabel="Pure state threshold", ylabel="Ratio $\\frac{Pure_t}{Mixed_t}$",
                 colors=[c.NT_COLOR, c.ASD_COLOR],
                 p_val_test_func=lambda x, y: st.ttest_rel(x, y)[1])

# %% effect of noise on population level slope
if utils.HETEROGENEITY_TO_N is None:
    _ = utils.get_effective_n_from_heterogeneity(0.1)
noise_level_list, retrieved_n = utils.HETEROGENEITY_TO_N
# plot
fig = plt.figure(figsize=pl.get_fig_size(1, 1.7))
ax: plt.Axes = fig.subplots()
ax.plot(noise_level_list, retrieved_n, color=c.ASD_COLOR)
pl.set_ax_labels(ax, "Increased heterogeneity in half activation point\nreduces slope (n) of population response",
                 "Heterogeneity level", "n")
pl.savefig(fig, "effect of noise on population slope", si=True, shift_x=-0.09, shift_y=1.05)
plt.close(fig)

# %% HGF
import scipy.io as scio

hgf_mat = scio.loadmat('hgf.mat', struct_as_record=False, squeeze_me=True)
asd_fit = hgf_mat["asd_fit"]
sim_fit = hgf_mat["sim_fit"]
alphas = hgf_mat["alphas"]
omegas = hgf_mat["omegas3"]


def tapas_sgm(x, a):
    return a / (1 + np.exp(-x))


def plot_param_trajectory(r, ax, title):
    t = np.ones_like(r.u)
    ts = np.concatenate([[0], np.cumsum(t)])
    ax.plot(ts, np.concatenate([[tapas_sgm(r.p_prc.mu_0[1], 1)], tapas_sgm(r.traj.mu[:, 1], 1)]),
            color='r', linewidth=4, label=r"$P(x_1=1)$")
    ax.scatter(0, tapas_sgm(r.p_prc.mu_0[1], 1), color="red", s=15)
    ax.plot(ts[1:np.argmax(r.u == 1)+1], r.u[:np.argmax(r.u == 1)], color=[0, 0.6, 0], label="$u$, Input", linewidth=3,
            linestyle=":")
    ax.plot(ts[np.argmax(r.u == 1)+1:], r.u[np.argmax(r.u == 1):], color=[0, 0.6, 0], linewidth=3, linestyle=":")
    ax.plot(ts[1:np.argmax(r.y == 1)+1], (((r.y - 0.5) * 1.05) + 0.5)[:np.argmax(r.y == 1)], color=[1, 0.7, 0],
            label="$y$, Response", linewidth=3, linestyle=":")
    ax.plot(ts[np.argmax(r.y == 1)+1:], (((r.y - 0.5) * 1.05) + 0.5)[np.argmax(r.y == 1):], color=[1, 0.7, 0],
            linewidth=3, linestyle=":")
    ax.plot(ts[1:], r.traj.wt[:, 0], color='k', linestyle=":", linewidth=3, label="Learning rate")
    ax.legend(loc='upper left', bbox_to_anchor=[0, 0.9, 0.2, 0.1], fontsize=15)
    ax.set_title(title, fontsize=25)
    ax.set_ylabel("Input and Response", fontsize=20)
    ax.set_xlabel("Trial number", fontsize=20)
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='y', which='major', labelsize=17)
    ax.tick_params(axis='y', which='minor', labelsize=17)


def plot_boxplots(arr, ax, title, ylabel):
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
    ax.set_xticklabels(["ASD", "NT"],fontweight='bold')
    ax.tick_params(axis='y', which='major', labelsize=17)
    ax.tick_params(axis='y', which='minor', labelsize=17)


fig, axes = plt.subplots(2, 2, figsize=pl.get_fig_size(1, 1.6))
axes = np.ravel(axes)

plot_param_trajectory(sim_fit, axes[0], "Base parameters fit")
plot_param_trajectory(asd_fit, axes[1], "High PU parameters fit")
plot_boxplots(alphas, axes[2], "Perceptual uncertainty", r"$\alpha$")
axes[2].set_ylim([0,axes[2].get_ylim()[1]])
plot_boxplots(omegas, axes[3], "Volatility estimates", r"$\omega_3$")

pl.savefig(fig, "hgf model", si=True, tight=True, shift_x=-0.15, shift_y=1.05, numbering_size=30)
