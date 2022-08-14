import numpy as np
import constants as c
import utils
import plotting as pl
from tqdm import tqdm
import scipy.stats


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
    asd_estimated_var[:, 0] = c.SR_PRIOR_VARIANCE
    nt_estimated_var = np.zeros_like(nt_var, dtype=np.float64)
    nt_estimated_var[:, 0] = c.SR_PRIOR_VARIANCE
    asd_resp_estimate = np.zeros_like(asd_var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    nt_resp_estimate = np.zeros_like(nt_var, dtype=np.float64)
    asd_resp_estimate[:, 0] = asd_resp.mean(-1)[:, 0]
    nt_resp_estimate[:, 0] = nt_resp.mean(-1)[:, 0]
    asd_kg = np.zeros_like(asd_var, dtype=np.float64)
    nt_kg = np.zeros_like(nt_var, dtype=np.float64)
    # run kalman filter
    print("running Kalman filter...")
    for i in tqdm(range(1, c.SR_NUM_STEPS)):
        utils.kalman_filter_step(i, asd_kg, asd_estimated_var, asd_var, asd_resp_estimate, asd_resp.mean(-1),
                                 c.SR_PRIOR_VARIANCE, c.SR_PERTURBATION_PROB)
        utils.kalman_filter_step(i, nt_kg, nt_estimated_var, nt_var, nt_resp_estimate, nt_resp.mean(-1),
                                 c.SR_PRIOR_VARIANCE, c.SR_PERTURBATION_PROB)

    asd_sig_estimate = np.zeros_like(asd_resp_estimate)
    nt_sig_estimate = np.zeros_like(nt_resp_estimate)
    for i in tqdm(range(c.SR_NUM_REPS)):
        asd_sig_estimate[i] = dense_sig[np.searchsorted(asd_dense_resp[i], asd_resp_estimate[i])]
        nt_sig_estimate[i] = dense_sig[np.searchsorted(nt_dense_resp[i], nt_resp_estimate[i])]

    # plot
    kalman_fig = pl.plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate)
    pl.savefig(kalman_fig, "slower updating in asd", shift_x=-0.06, shift_y=1.03, tight=False,
               ignore=kalman_fig.get_axes())
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
        pl.savefig(shift_fig, save_name="ASD to NT shift function of reaction time to abrupt change",
                   ignore=shift_fig.get_axes(), si=True)
        p, permutation_fig = utils.permutation_test(asd_time_to_09_sig, nt_time_to_09_sig,
                                                    lambda x, y: x.var() - y.var(), alternative="greater", plot=True,
                                                    center_dists=True, hist_color=c.EI_COLOR, line_color=c.ASD_COLOR)
        print(
            f"Reaction time variance, ASD > NT hypothesis permutation test, diff={asd_time_to_09_sig.var() - nt_time_to_09_sig.var()}, p = {utils.num2print(p)}")
        pl.savefig(permutation_fig, "Reaction time variance permutation test", ignore=permutation_fig.get_axes(),
                   si=True)
    else:
        var_stat, var_p_val = utils.f_test(asd_time_to_09_sig, nt_time_to_09_sig, "greater")
        print(f"Variance equality F-test, F: {var_stat:.4g}, p-value: {var_p_val:.4g}")


if __name__ == '__main__':
    simulate_signal_change_tracking_update()
