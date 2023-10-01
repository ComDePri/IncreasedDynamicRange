import os

import numpy as np
import constants as c
import idr_utils as utils
import plotting as pl
from tqdm import tqdm
import scipy.stats
from data_fitting import TappingData, permutation_fit_to_group_dynamics
import warnings

warnings.simplefilter('ignore', RuntimeWarning)


def prepare_signal():
    print("generating signal...")
    # create noisy signal
    signal = np.zeros((c.SR_NUM_REPS, c.SR_NUM_STEPS, c.SR_N_NEURONS))
    change_timepoint = round(max(c.SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
    signal[:, :change_timepoint, :] = c.SR_SIG_MIN
    signal[:, change_timepoint:, :] = c.SR_SIG_MAX
    signal += np.random.normal(scale=c.SR_SIG_SIGMA, size=signal.shape)
    signal[signal < 0] = 0
    signal[signal > 1] = 1
    return change_timepoint, signal


def get_populations():
    print("generating neuronal populations...")
    # create km of ASD and NT, same across repeats
    asd_km = 0.5 + c.SR_ASD_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))
    nt_km = 0.5 + c.SR_NT_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))
    ei_km = 0.5 + c.SR_NT_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))
    return asd_km, ei_km, nt_km


def get_signal_encoding(asd_km, nt_km, signal, si):
    print("calculating neuronal responses...")
    if si:
        gain_func = utils.logistic_func
        n = c.SI_SR_N
    else:
        gain_func = utils.ei_gain_func
        n = c.SR_N
    asd_resp = gain_func(signal, n, asd_km)
    nt_resp = gain_func(signal, n, nt_km)
    ei_resp = gain_func(signal, n, nt_km, c.EI_NU)
    return asd_resp, ei_resp, nt_resp


def get_encoding_variance(asd_resp, ei_resp, nt_resp):
    print("calculating neuronal response variances...")
    asd_var = asd_resp.var(-1) / c.SR_N_NEURONS
    nt_var = nt_resp.var(-1) / c.SR_N_NEURONS
    ei_var = ei_resp.var(-1) / c.SR_N_NEURONS
    return asd_var, ei_var, nt_var


def initialize_kalman_filter_arrays(asd_resp, asd_var, ei_resp, ei_var, nt_resp, nt_var):
    print("initializing Kalman filter variables...")
    asd_estimated_var = np.zeros_like(asd_var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    asd_estimated_var[:, 0] = c.SR_PRIOR_VARIANCE
    nt_estimated_var = np.zeros_like(nt_var, dtype=np.float64)
    nt_estimated_var[:, 0] = c.SR_PRIOR_VARIANCE
    ei_estimated_var = np.zeros_like(ei_var, dtype=np.float64)
    ei_estimated_var[:, 0] = c.SR_PRIOR_VARIANCE
    asd_resp_estimate = np.zeros_like(asd_var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    nt_resp_estimate = np.zeros_like(nt_var, dtype=np.float64)
    ei_resp_estimate = np.zeros_like(ei_var, dtype=np.float64)
    asd_resp_estimate[:, 0] = asd_resp.mean(-1)[:, 0]
    nt_resp_estimate[:, 0] = nt_resp.mean(-1)[:, 0]
    ei_resp_estimate[:, 0] = ei_resp.mean(-1)[:, 0]
    asd_kg = np.zeros_like(asd_var, dtype=np.float64)
    nt_kg = np.zeros_like(nt_var, dtype=np.float64)
    ei_kg = np.zeros_like(ei_var, dtype=np.float64)
    return asd_estimated_var, asd_kg, asd_resp_estimate, ei_estimated_var, ei_kg, ei_resp_estimate, nt_estimated_var, nt_kg, nt_resp_estimate


def run_kalman_filter(si):
    print("=========================================\n"
          "==== Kalman filter update simulation ====\n"
          "=========================================")
    np.random.seed(c.SEED)
    change_timepoint, signal = prepare_signal()
    asd_km, ei_km, nt_km = get_populations()
    # get responses
    asd_resp, ei_resp, nt_resp = get_signal_encoding(asd_km, nt_km, signal, si)
    # extract variances per step per repeat
    asd_var, ei_var, nt_var = get_encoding_variance(asd_resp, ei_resp, nt_resp)
    # run Kalman filter on the measured responses
    # initialize estimates and confidences:
    asd_estimated_var, asd_kg, asd_resp_estimate, ei_estimated_var, ei_kg, ei_resp_estimate, nt_estimated_var, nt_kg, nt_resp_estimate = initialize_kalman_filter_arrays(
        asd_resp, asd_var, ei_resp, ei_var, nt_resp, nt_var)
    # run kalman filter
    print("running Kalman filter...")
    for i in tqdm(range(1, c.SR_NUM_STEPS)):
        utils.kalman_filter_step(i, asd_kg, asd_estimated_var, asd_var, asd_resp_estimate, asd_resp.mean(-1),
                                 c.SR_PRIOR_VARIANCE, c.SR_PERTURBATION_PROB)
        utils.kalman_filter_step(i, nt_kg, nt_estimated_var, nt_var, nt_resp_estimate, nt_resp.mean(-1),
                                 c.SR_PRIOR_VARIANCE, c.SR_PERTURBATION_PROB)
        utils.kalman_filter_step(i, ei_kg, ei_estimated_var, ei_var, ei_resp_estimate, ei_resp.mean(-1),
                                 c.SR_PRIOR_VARIANCE, c.SR_PERTURBATION_PROB)

    asd_sig_estimate, ei_sig_estimate, nt_sig_estimate = decode_signal(asd_km, asd_resp_estimate, ei_km,
                                                                       ei_resp_estimate, nt_km, nt_resp_estimate, si)
    return asd_sig_estimate, change_timepoint, ei_sig_estimate, nt_sig_estimate, signal


def decode_signal(asd_km, asd_resp_estimate, ei_km, ei_resp_estimate, nt_km, nt_resp_estimate, si):
    print("Decoding responses...")
    asd_sig_estimate = np.zeros_like(asd_resp_estimate)
    nt_sig_estimate = np.zeros_like(nt_resp_estimate)
    ei_sig_estimate = np.zeros_like(ei_resp_estimate)
    dense_sig = np.linspace(0, 1, 5000)
    if si:
        gain_func = utils.logistic_func
        n = c.SI_SR_N
    else:
        gain_func = utils.ei_gain_func
        n = c.SR_N
    asd_dense_resp = np.squeeze(gain_func(dense_sig[None, :, None], n, asd_km).mean(-1))
    nt_dense_resp = np.squeeze(gain_func(dense_sig[None, :, None], n, nt_km).mean(-1))
    ei_dense_resp = np.squeeze(gain_func(dense_sig[None, :, None], n, ei_km, c.EI_NU).mean(-1))
    for i in tqdm(range(c.SR_NUM_REPS)):
        asd_sig_estimate[i] = dense_sig[np.searchsorted(asd_dense_resp[i], asd_resp_estimate[i])]
        nt_sig_estimate[i] = dense_sig[np.searchsorted(nt_dense_resp[i], nt_resp_estimate[i])]
        ei_sig_estimate[i] = dense_sig[np.searchsorted(ei_dense_resp[i], ei_resp_estimate[i])]
    return asd_sig_estimate, ei_sig_estimate, nt_sig_estimate


def fit_tapping_data():
    print("Fitting tapping data...")
    fitting_kwargs = dict(prior_var=0.1, perturb_prob=1e-6, base_n=20, n_neurons=200, perceptual_noise=0.03,
                          fit_scale_factor=50)
    asd_data = TappingData("ASD.mat", **fitting_kwargs)
    nt_data = TappingData("NT.mat", **fitting_kwargs)
    asd_data.fit_to_group_dynamics()
    nt_data.fit_to_group_dynamics()
    print(
        f"Fitted Hill coefficients:\nASD = {np.nanmean(asd_data.fitted_n):.2g}$\pm${np.nanstd(asd_data.fitted_n) / len(asd_data):.2g}"
        f",\nNT = {np.nanmean(nt_data.fitted_n):.2g}$\pm${np.nanstd(nt_data.fitted_n) / len(nt_data):.2g}"
    )
    ei_data = TappingData("ASD.mat", **fitting_kwargs)
    ei_data.fit_ei_to_group_dynamics(min_n=np.nanmean(nt_data.fitted_n), nu_range=[0.1, 1.])
    print(
        f"Fitted Hill coefficients:\nASD = {np.nanmean(asd_data.fitted_n):.2g}$\pm${np.nanstd(asd_data.fitted_n) / len(asd_data):.2g}"
        f",\nNT = {np.nanmean(nt_data.fitted_n):.2g}$\pm${np.nanstd(nt_data.fitted_n) / len(nt_data):.2g}"
        f",\nEI = {np.nanmean(ei_data.fitted_n):.2g}$\pm${np.nanstd(ei_data.fitted_n):.2g}, "
        f"nu={np.nanmean(ei_data.fitted_nu):.2g}$\pm${np.nanstd(ei_data.fitted_nu) / len(asd_data):.2g}\n")
    print("Mann Whitney U test ASD < NT Hill coefficients, results:")
    print(scipy.stats.mannwhitneyu(asd_data.fitted_n[~np.isnan(asd_data.fitted_n)].ravel(),
                                   nt_data.fitted_n[~np.isnan(nt_data.fitted_n)].ravel(), alternative="less"))
    # print("Bootstrapping group n:")
    # asd_data.bootstrap_fit_to_group_dynamics()
    # nt_data.bootstrap_fit_to_group_dynamics()
    # print(f"Bootstrap SE: ASD = {np.nanmean(np.nanstd(asd_data.boot_fitted_n, axis=0)):.2g}, "
    #       f"NT = {np.nanmean(np.nanstd(nt_data.boot_fitted_n, axis=0)):.2g}")
    # permutation_n = permutation_fit_to_group_dynamics(asd_data, nt_data)
    # print(f"Permutation test p-value: "
    #       f"{np.mean(np.nanmean(nt_data.fitted_n) - np.nanmean(asd_data.fitted_n) < permutation_n):.3g}")
    return asd_data, nt_data, ei_data


def rt_difference_tests(asd_sig_estimate, change_timepoint, ei_sig_estimate, nt_sig_estimate, si):
    asd_time_to_09_sig = np.argmin(np.abs(asd_sig_estimate[change_timepoint:] - (c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX)),
                                   1) - change_timepoint
    nt_time_to_09_sig = np.argmin(np.abs(nt_sig_estimate[change_timepoint:] - (c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX)),
                                  1) - change_timepoint
    ei_time_to_09_sig = np.argmin(np.abs(ei_sig_estimate[change_timepoint:] - (c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX)),
                                  1) - change_timepoint
    savename_prefix = "logistic_func/" if si else ""
    print(f"ASD RT, mean: {asd_time_to_09_sig.mean()}, STD: {asd_time_to_09_sig.std()}\n",
          f"NT RT, mean: {nt_time_to_09_sig.mean()}, STD: {nt_time_to_09_sig.std()}\n",
          f"EI RT, mean: {ei_time_to_09_sig.mean()}, STD: {ei_time_to_09_sig.std()}\n")
    print(
        f"Wilcoxon signed-rank test for ASD and NT time to 0.9 of signal after change: {scipy.stats.wilcoxon(asd_time_to_09_sig, nt_time_to_09_sig)}")
    print(
        f"Wilcoxon signed-rank test for ASD and EI time to 0.9 of signal after change: {scipy.stats.wilcoxon(asd_time_to_09_sig, ei_time_to_09_sig)}")
    print(
        f"Wilcoxon signed-rank test for EI and NT time to 0.9 of signal after change: {scipy.stats.wilcoxon(ei_time_to_09_sig, nt_time_to_09_sig)}")
    print("Last Km normality test: ")
    # Check normality assumption to use correct statistics.
    _, asd_normal_p = scipy.stats.shapiro(nt_time_to_09_sig)
    _, nt_normal_p = scipy.stats.shapiro(asd_time_to_09_sig)
    _, ei_normal_p = scipy.stats.shapiro(ei_time_to_09_sig)
    print(f"ASD normality: {asd_normal_p}, NT normality: {nt_normal_p}, EI normality: {ei_normal_p}")
    if nt_normal_p < 0.05 or asd_normal_p < 0.05 and ei_normal_p < 0.05:
        print("Can't calculate F-test as normality assumption fails!\n"
              "Calculating Shift-function & permutation test for variance instead...")
        shift_fig = pl.plot_shift_func(asd_time_to_09_sig, nt_time_to_09_sig,
                                       "Reaction time to abrupt change, ASD to NT")
        pl.savefig(shift_fig, save_name=savename_prefix + "ASD to NT shift function of reaction time to abrupt change",
                   ignore=shift_fig.get_axes(), si=True)
        shift_fig = pl.plot_shift_func(asd_time_to_09_sig, ei_time_to_09_sig,
                                       savename_prefix + "Reaction time to abrupt change, ASD to EI")
        pl.savefig(shift_fig, save_name="ASD to EI shift function of reaction time to abrupt change",
                   ignore=shift_fig.get_axes(), si=True)
        shift_fig = pl.plot_shift_func(nt_time_to_09_sig, ei_time_to_09_sig,
                                       "Reaction time to abrupt change, NT to EI")
        pl.savefig(shift_fig, save_name=savename_prefix + "NT to EI shift function of reaction time to abrupt change",
                   ignore=shift_fig.get_axes(), si=True)

        p, permutation_fig = utils.permutation_test(asd_time_to_09_sig, nt_time_to_09_sig,
                                                    lambda x, y: x.var() - y.var(), alternative="greater", plot=True,
                                                    center_dists=True, hist_color=c.EI_COLOR, line_color=c.ASD_COLOR)
        print(
            f"Reaction time variance, ASD > NT hypothesis permutation test, diff={asd_time_to_09_sig.var() - nt_time_to_09_sig.var()}, p = {utils.num2print(p)}")
        pl.savefig(permutation_fig, savename_prefix + "Reaction time variance permutation test ASD-NT",
                   ignore=permutation_fig.get_axes(),
                   si=True)

        p, permutation_fig = utils.permutation_test(asd_time_to_09_sig, ei_time_to_09_sig,
                                                    lambda x, y: x.var() - y.var(), alternative="greater", plot=True,
                                                    center_dists=True, hist_color=c.EI_COLOR, line_color=c.ASD_COLOR)
        print(
            f"Reaction time variance, ASD > EI hypothesis permutation test, diff={asd_time_to_09_sig.var() - ei_time_to_09_sig.var()}, p = {utils.num2print(p)}")
        pl.savefig(permutation_fig, savename_prefix + "Reaction time variance permutation test ASD-EI",
                   ignore=permutation_fig.get_axes(),
                   si=True)

        p, permutation_fig = utils.permutation_test(ei_time_to_09_sig, nt_time_to_09_sig,
                                                    lambda x, y: x.var() - y.var(), alternative="two.sided", plot=True,
                                                    center_dists=True, hist_color=c.EI_COLOR, line_color=c.ASD_COLOR)
        print(
            f"Reaction time variance, NT = EI hypothesis permutation test, diff={ei_time_to_09_sig.var() - nt_time_to_09_sig.var()}, p = {utils.num2print(p)}")
        pl.savefig(permutation_fig, savename_prefix + "Reaction time variance permutation test EI-NT",
                   ignore=permutation_fig.get_axes(),
                   si=True)
    else:
        var_stat, var_p_val = utils.f_test(asd_time_to_09_sig, nt_time_to_09_sig, "greater")
        print(f"Variance equality F-test, ASD-NT, F: {var_stat:.4g}, p-value: {var_p_val:.4g}")

        var_stat, var_p_val = utils.f_test(asd_time_to_09_sig, ei_time_to_09_sig, "greater")
        print(f"Variance equality F-test, ASD-EI, F: {var_stat:.4g}, p-value: {var_p_val:.4g}")

        var_stat, var_p_val = utils.f_test(ei_time_to_09_sig, nt_time_to_09_sig, "two.sided")
        print(f"Variance equality F-test, EI-NT, F: {var_stat:.4g}, p-value: {var_p_val:.4g}")


def single_subject_tapping_fit(asd_data, nt_data, asd_ei_data):
    print("Fitting single subjects...")
    # tapping data fit SI plots
    for i, asd_subj in enumerate(tqdm(asd_data, desc="ASD subjects")):
        asd_subj.fit_n()
        fig = asd_subj.plot_change_dynamics(c=c.ASD_COLOR)
        pl.savefig(fig, f"tapping_figures/asd_subject_{str(i).zfill(3)}", ignore=fig.get_axes(), si=True)
        pl.plt.close()
    for i, nt_subj in enumerate(tqdm(nt_data, desc="NT subjects")):
        nt_subj.fit_n()
        fig = nt_subj.plot_change_dynamics(c=c.NT_COLOR)
        pl.savefig(fig, f"tapping_figures/nt_subject_{str(i).zfill(3)}", ignore=fig.get_axes(), si=True)
        pl.plt.close()
    for i, ei_subj in enumerate(tqdm(asd_ei_data, desc="ASD EI subjects")):
        ei_subj.fit_ei_n_per_block()
        fig = ei_subj.plot_change_dynamics(c=c.EI_COLOR)
        pl.savefig(fig, f"tapping_figures/asd_ei_subject_{str(i).zfill(3)}", ignore=fig.get_axes(), si=True)
        pl.plt.close()

    # show different tracking times per subject
    def get_track_time(group_dynamics):
        acc = group_dynamics[:, :, 0, 3:] < 500
        dec = group_dynamics[:, :, 1, 3:] > 500
        acc_ok = acc.any(axis=-1)
        dec_ok = dec.any(axis=-1)
        acc = np.argmax(acc, axis=-1).astype(float)
        dec = np.argmax(dec, axis=-1).astype(float)
        acc[~acc_ok] = np.nan
        dec[~dec_ok] = np.nan
        return acc[:, 2:], dec[:, 2:]

    asd_hill_coefs = asd_data.get_median_hill_coefficients()
    nt_hill_coefs = nt_data.get_median_hill_coefficients()
    ei_hill_coefs = asd_ei_data.get_median_hill_coefficients()
    print(f"Single-subject Hill coefficients\n"
          f"ASD: #={asd_hill_coefs.size}, {np.nanmean(asd_hill_coefs):.2g}$$\\pm$${np.nanstd(asd_hill_coefs):.2g}\n"
          f"NT: #={nt_hill_coefs.size}, {np.nanmean(nt_hill_coefs):.2g}$$\\pm$${np.nanstd(nt_hill_coefs):.2g}\n"
          f"ASD EI: #={ei_hill_coefs.size}, {np.nanmean(ei_hill_coefs):.2g}$$\\pm$${np.nanstd(ei_hill_coefs):.2g}")

    print("Mann-Whitney test for ASD and NT coefficients:")
    print(scipy.stats.mannwhitneyu(nt_hill_coefs, asd_hill_coefs))
    asd_accelerating_change_idx, asd_decelerating_change_idx = get_track_time(
        np.stack(asd_data.get_change_dynamics()[0]))
    nt_accelerating_change_idx, nt_decelerating_change_idx = get_track_time(np.stack(nt_data.get_change_dynamics()[0]))
    print(f"% no track accelerating, ASD: {100 * np.mean(np.isnan(asd_accelerating_change_idx)):.2f}%")
    print(f"% no track decelerating, ASD: {100 * np.mean(np.isnan(asd_decelerating_change_idx)):.2f}%")
    print(f"% no track accelerating, NT: {100 * np.mean(np.isnan(nt_accelerating_change_idx)):.2f}%")
    print(f"% no track decelerating, NT: {100 * np.mean(np.isnan(nt_decelerating_change_idx)):.2f}%")
    nt_tau = np.concatenate([nt_decelerating_change_idx.ravel(), nt_accelerating_change_idx.ravel()])
    asd_tau = np.concatenate([asd_decelerating_change_idx.ravel(), asd_accelerating_change_idx.ravel()])
    nt_tau = nt_tau[~np.isnan(nt_tau)]
    asd_tau = asd_tau[~np.isnan(asd_tau)]
    fig = pl.plot_tapping_tracking_delays(asd_tau, nt_tau)
    pl.savefig(fig, "Tracking delays", si=True, ignore=fig.get_axes())
    print("Mann-Whitney test for ASD and NT delays:")
    print(scipy.stats.mannwhitneyu(nt_tau, asd_tau))


def simulate_signal_change_tracking_update(si=False):
    asd_sig_estimate, change_timepoint, ei_sig_estimate, nt_sig_estimate, signal = run_kalman_filter(si)

    # fit tapping data
    if not si:
        asd_data, nt_data, ei_data = fit_tapping_data()
        print(
            "Model fit loss:\n"
            f"ASD:\n{np.nanmean(asd_data.simulated_dynamics_loss)}\n"
            f"NT:\n{np.nanmean(nt_data.simulated_dynamics_loss)}\n"
            f"ASD EI:\n{np.nanmean(ei_data.simulated_dynamics_loss)}\n"
        )
        kalman_fig = pl.plot_slower_updating(signal, asd_sig_estimate, nt_sig_estimate, asd_data, nt_data)
        single_subject_tapping_fit(asd_data, nt_data, ei_data)
        fig = pl.plot_tapping_fit(asd_data, color=c.ASD_COLOR, title="ASD group fit")
        pl.savefig(fig, "ASD group fit", si=True, ignore=fig.get_axes())
        fig = pl.plot_tapping_fit(nt_data, color=c.NT_COLOR, title="NT group fit")
        pl.savefig(fig, "NT group fit", si=True, ignore=fig.get_axes())
        fig = pl.plot_tapping_fit(ei_data, color=c.EI_COLOR, title="E/I group fit")
        pl.savefig(fig, "EI group fit", si=True, ignore=fig.get_axes())
        pl.plt.close('all')
        pl.combine_plots(["SI figures/ASD group fit.png", "SI figures/NT group fit.png", "SI figures/EI group fit.png"],
                         "SI figures/tapping group fit comparisons.png")

    else:
        kalman_fig = pl.plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate)
    savename = "slower updating in asd"
    if si:
        savename = os.path.join("logistic_func", savename)
    pl.savefig(kalman_fig, "slower updating in asd", shift_x=-0.15, shift_y=1.1, tight=False, numbering_size=30,
               figure_coord_labels=[(0.05, 0.95), (0.51, 0.95), (0.05, 0.475), (0.51, 0.475)], si=si)

    kalman_fig = pl.plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate, ei_sig_estimate)
    pl.savefig(kalman_fig, savename + " with ei", shift_x=-0.06, shift_y=1.03, tight=False, si=True,
               ignore=kalman_fig.get_axes())

    # statistics - time to 0.5/0.9 of the signal change
    rt_difference_tests(asd_sig_estimate, change_timepoint, ei_sig_estimate, nt_sig_estimate, si)


if __name__ == '__main__':
    simulate_signal_change_tracking_update()
    pl.plt.close('all')
