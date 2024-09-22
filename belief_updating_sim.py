import os
import warnings

import constants as c
import idr_utils as utils
import numpy as np
import plotting as pl
import scipy.stats
from data_fitting import TappingData, permutation_fit_to_group_dynamics
from tqdm import tqdm

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


def fit_tapping_data(boot=False,perm=False):
    print("Fitting tapping data...")
    fitting_kwargs = dict(prior_var=0.1, perturb_prob=1e-6, base_n=20, n_neurons=200, perceptual_noise=0.03,
                          fit_scale_factor=50, half_range=0.3)
    asd_data = TappingData("ASD.mat", **fitting_kwargs)
    nt_data = TappingData("NT.mat", **fitting_kwargs)
    asd_data.fit_to_group_dynamics()
    nt_data.fit_to_group_dynamics()
    print(
        rf"Fitted Hill coefficients:\nASD = {np.nanmean(asd_data.fitted_n):.2g}$\pm${np.nanstd(asd_data.fitted_n) / np.sqrt(len(asd_data)):.2g}"
        rf",\nNT = {np.nanmean(nt_data.fitted_n):.2g}$\pm${np.nanstd(nt_data.fitted_n) / np.sqrt(len(nt_data)):.2g}"
    )
    ei_data = TappingData("ASD.mat", **fitting_kwargs)
    ei_data.fit_ei_to_group_dynamics(min_n=np.nanmean(nt_data.fitted_n), nu_range=[0.1, 1.])
    print(
        rf"Fitted Hill coefficients:\nASD = {np.nanmean(asd_data.fitted_n):.2g}$\pm${np.nanstd(asd_data.fitted_n) / np.sqrt(len(asd_data)):.2g}"
        rf",\nNT = {np.nanmean(nt_data.fitted_n):.2g}$\pm${np.nanstd(nt_data.fitted_n) / np.sqrt(len(nt_data)):.2g}"
        rf",\nEI = {np.nanmean(ei_data.fitted_n):.2g}$\pm${np.nanstd(ei_data.fitted_n) / np.sqrt(len(asd_data)):.2g}, "
        rf"nu={np.nanmean(ei_data.fitted_nu):.2g}$\pm${np.nanstd(ei_data.fitted_nu) / np.sqrt(len(asd_data)):.2g}\n")
    print("Mann Whitney U test ASD < NT Hill coefficients, results:")
    print(scipy.stats.mannwhitneyu(asd_data.fitted_n[~np.isnan(asd_data.fitted_n)].ravel(),
                                   nt_data.fitted_n[~np.isnan(nt_data.fitted_n)].ravel(), alternative="less"))
    if boot:
        print("Bootstrapping group n:")
        asd_data.bootstrap_fit_to_group_dynamics()
        nt_data.bootstrap_fit_to_group_dynamics()
        print(f"Bootstrap SE: ASD = {np.nanmean(np.nanstd(asd_data.boot_fitted_n, axis=0)):.2g}, "
              f"NT = {np.nanmean(np.nanstd(nt_data.boot_fitted_n, axis=0)):.2g}")
    if perm:
        permutation_n = permutation_fit_to_group_dynamics(asd_data, nt_data)
        print(f"Permutation test p-value: "
              f"{np.mean(np.nanmean(nt_data.fitted_n) - np.nanmean(asd_data.fitted_n) < permutation_n):.3g}")
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
                                       "Reaction time to abrupt change, IDR to NDR")
        pl.savefig(shift_fig, save_name=savename_prefix + "ASD to NT shift function of reaction time to abrupt change",
                   ignore=shift_fig.get_axes(), si=True)
        shift_fig = pl.plot_shift_func(asd_time_to_09_sig, ei_time_to_09_sig,
                                       savename_prefix + "Reaction time to abrupt change, IDR to EI")
        pl.savefig(shift_fig, save_name=savename_prefix + "ASD to EI shift function of reaction time to abrupt change",
                   ignore=shift_fig.get_axes(), si=True)
        shift_fig = pl.plot_shift_func(nt_time_to_09_sig, ei_time_to_09_sig,
                                       "Reaction time to abrupt change, NDR to EI")
        pl.savefig(shift_fig, save_name=savename_prefix + "NT to EI shift function of reaction time to abrupt change",
                   ignore=shift_fig.get_axes(), si=True)

        p, permutation_fig = utils.permutation_test(asd_time_to_09_sig, nt_time_to_09_sig,
                                                    lambda x, y: x.var() - y.var(), alternative="greater", plot=True,
                                                    center_dists=True, hist_color=c.EI_COLOR, line_color=c.ASD_COLOR)
        print(
            f"Reaction time variance, IDR > NDR hypothesis permutation test, diff={asd_time_to_09_sig.var() - nt_time_to_09_sig.var()}, p = {utils.num2print(p)}")
        pl.savefig(permutation_fig, savename_prefix + "Reaction time variance permutation test ASD-NT",
                   ignore=permutation_fig.get_axes(),
                   si=True)

        p, permutation_fig = utils.permutation_test(asd_time_to_09_sig, ei_time_to_09_sig,
                                                    lambda x, y: x.var() - y.var(), alternative="greater", plot=True,
                                                    center_dists=True, hist_color=c.EI_COLOR, line_color=c.ASD_COLOR)
        print(
            f"Reaction time variance, IDR > EI hypothesis permutation test, diff={asd_time_to_09_sig.var() - ei_time_to_09_sig.var()}, p = {utils.num2print(p)}")
        pl.savefig(permutation_fig, savename_prefix + "Reaction time variance permutation test ASD-EI",
                   ignore=permutation_fig.get_axes(),
                   si=True)

        p, permutation_fig = utils.permutation_test(ei_time_to_09_sig, nt_time_to_09_sig,
                                                    lambda x, y: x.var() - y.var(), alternative="two.sided", plot=True,
                                                    center_dists=True, hist_color=c.EI_COLOR, line_color=c.ASD_COLOR)
        print(
            f"Reaction time variance, NDR = EI hypothesis permutation test, diff={ei_time_to_09_sig.var() - nt_time_to_09_sig.var()}, p = {utils.num2print(p)}")
        pl.savefig(permutation_fig, savename_prefix + "Reaction time variance permutation test EI-NT",
                   ignore=permutation_fig.get_axes(),
                   si=True)
    else:
        var_stat, var_p_val = utils.f_test(asd_time_to_09_sig, nt_time_to_09_sig, "greater")
        print(f"Variance equality F-test, IDR-NDR, F: {var_stat:.4g}, p-value: {var_p_val:.4g}")

        var_stat, var_p_val = utils.f_test(asd_time_to_09_sig, ei_time_to_09_sig, "greater")
        print(f"Variance equality F-test, IDR-EI, F: {var_stat:.4g}, p-value: {var_p_val:.4g}")

        var_stat, var_p_val = utils.f_test(ei_time_to_09_sig, nt_time_to_09_sig, "two.sided")
        print(f"Variance equality F-test, EI-NDR, F: {var_stat:.4g}, p-value: {var_p_val:.4g}")


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
            f"ASD:\n{utils.mse_bic(np.nanmean(asd_data.simulated_dynamics_loss), 1, len(asd_data))}\n"
            f"NT:\n{utils.mse_bic(np.nanmean(nt_data.simulated_dynamics_loss), 1, len(nt_data))}\n"
            f"ASD EI:\n{utils.mse_bic(np.nanmean(ei_data.simulated_dynamics_loss), 2, len(ei_data))}\n"
        )
        kalman_fig = pl.plot_slower_updating(signal, asd_sig_estimate, nt_sig_estimate, asd_data, nt_data)
        single_subject_tapping_fit(asd_data, nt_data, ei_data)
        fig = pl.plot_tapping_fit(asd_data, color=c.ASD_COLOR, title="ASD group fit")
        pl.savefig(fig, "ASD group fit", si=True, ignore=fig.get_axes())
        fig = pl.plot_tapping_fit(nt_data, color=c.NT_COLOR, title="NT group fit")
        pl.savefig(fig, "NT group fit", si=True, ignore=fig.get_axes())
        fig = pl.plot_tapping_fit(ei_data, color=c.EI_COLOR, title="ASD group fit decreased inhibition")
        pl.savefig(fig, "EI group fit", si=True, ignore=fig.get_axes())
        pl.plt.close('all')
        pl.combine_plots(["SI figures/ASD group fit.pdf", "SI figures/NT group fit.pdf", "SI figures/EI group fit.pdf"],
                         "SI figures/tapping group fit comparisons.pdf")

    else:
        kalman_fig = pl.plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate)
    savename = "slower updating in asd"
    if si:
        savename = os.path.join("logistic_func", savename)
        pl.savefig(kalman_fig, "slower updating in asd", ignore=[kalman_fig.get_axes()], tight=False, si=si)
    else:
        pl.savefig(kalman_fig, "slower updating in asd", shift_x=-0.15, shift_y=1.1, tight=False, numbering_size=30,
                   figure_coord_labels=[(0.05, 0.95), (0.51, 0.95), (0.05, 0.475), (0.51, 0.475)], si=si)

    kalman_fig = pl.plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate, ei_sig_estimate)
    pl.savefig(kalman_fig, savename + " with ei", shift_x=-0.06, shift_y=1.03, tight=False, si=True,
               ignore=kalman_fig.get_axes())

    # statistics - time to 0.5/0.9 of the signal change
    rt_difference_tests(asd_sig_estimate, change_timepoint, ei_sig_estimate, nt_sig_estimate, si)

    # effect size as a function of n
    effect_size_analysis()


def power_analysis(het_het=0.1):
    n_list = np.array([8, 15])
    signal = np.zeros((n_list.size, 200, c.SR_NUM_STEPS, c.SR_N_NEURONS))
    change_timepoint = round(max(c.SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
    signal[:, :, :change_timepoint, :] = c.SR_SIG_MIN
    signal[:, :, change_timepoint:, :] = c.SR_SIG_MAX
    signal += np.random.normal(scale=c.SR_SIG_SIGMA, size=signal.shape)
    signal[signal < 0] = 0
    signal[signal > 1] = 1
    heterogeneities = np.zeros_like(n_list, dtype=float)
    for i, n in enumerate(tqdm(n_list, desc="Calculating heterogeneity levels")):
        heterogeneities[i] = utils.get_heterogeneity_from_n(n, c.SR_N, c.SR_N_NEURONS)
    hets = (heterogeneities[:, None, None, None] + np.random.normal(0, het_het, size=(signal.shape[:-2] + (1, 1))))
    km = 0.5 + hets * np.random.uniform(-1, 1, size=(n_list.size, 200, 1, c.SR_N_NEURONS))
    asd_hets = utils.get_effective_n_from_heterogeneity(hets[0], 20)
    resp = utils.ei_gain_func(signal, c.SR_N, km)
    var = resp.var(-1) / c.SR_N_NEURONS
    estimated_var = np.zeros_like(var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    estimated_var[..., 0] = c.SR_PRIOR_VARIANCE
    resp_estimate = np.zeros_like(var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    resp_estimate[..., 0] = resp.mean(-1)[..., 0]
    kg = np.zeros_like(var, dtype=np.float64)
    print("running Kalman filter...")
    for i in tqdm(range(1, c.SR_NUM_STEPS)):
        utils.kalman_filter_step(i, kg, estimated_var, var, resp_estimate, resp.mean(-1),
                                 c.SR_PRIOR_VARIANCE, 5e-4)
    sig_estimate = np.zeros_like(resp_estimate)
    dense_sig = np.linspace(0, 1, 5000)
    for n in tqdm(range(n_list.size)):
        dense_resp = np.squeeze(utils.hill_func(dense_sig[None, :, None], c.SR_N, km[n]).mean(-1))
        for i in range(200):
            sig_estimate[n, i] = dense_sig[np.searchsorted(dense_resp[i], resp_estimate[n, i])]
    time_to_09_sig = np.argmin(np.abs(sig_estimate[..., change_timepoint:] - (c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX)),
                               -1)
    pop_size, power_ana = utils.power_analysis(time_to_09_sig[0], time_to_09_sig[1], scipy.stats.mannwhitneyu,
                                               alternative="greater")
    fig = pl.plot_power_analysis(pop_size, power_ana, "Reaction time to abrupt change")
    pl.savefig(fig, "Reaction time power analysis", si=True, ignore=fig.get_axes())
    return pop_size, power_ana


def effect_size_analysis():
    n_space = np.linspace(3, 16, 25)
    signal = np.zeros((n_space.size, 200, c.SR_NUM_STEPS, c.SR_N_NEURONS))
    change_timepoint = round(max(c.SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
    signal[:, :, :change_timepoint, :] = c.SR_SIG_MIN
    signal[:, :, change_timepoint:, :] = c.SR_SIG_MAX
    signal += np.random.normal(scale=c.SR_SIG_SIGMA, size=signal.shape)
    signal[signal < 0] = 0
    signal[signal > 1] = 1
    heterogeneities = np.zeros_like(n_space, dtype=float)
    for i, n in enumerate(tqdm(n_space, desc="Calculating heterogeneity levels")):
        heterogeneities[i] = utils.get_heterogeneity_from_n(n, c.SR_N, c.SR_N_NEURONS)
    km = 0.5 + heterogeneities[:, None, None, None] * np.random.uniform(-1, 1, size=(
        n_space.size, 200, 1, c.SR_N_NEURONS))
    resp = utils.ei_gain_func(signal, c.SR_N, km)
    var = resp.var(-1) / c.SR_N_NEURONS
    estimated_var = np.zeros_like(var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    estimated_var[..., 0] = c.SR_PRIOR_VARIANCE
    resp_estimate = np.zeros_like(var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
    resp_estimate[..., 0] = resp.mean(-1)[..., 0]
    kg = np.zeros_like(var, dtype=np.float64)
    print("running Kalman filter...")
    for i in tqdm(range(1, c.SR_NUM_STEPS)):
        utils.kalman_filter_step(i, kg, estimated_var, var, resp_estimate, resp.mean(-1),
                                 c.SR_PRIOR_VARIANCE, c.SR_PERTURBATION_PROB)
    sig_estimate = np.zeros_like(resp_estimate)
    dense_sig = np.linspace(0, 1, 5000)
    for n in tqdm(range(n_space.size)):
        dense_resp = np.squeeze(utils.hill_func(dense_sig[None, :, None], c.SR_N, km[n]).mean(-1))
        for i in range(200):
            sig_estimate[n, i] = dense_sig[np.searchsorted(dense_resp[i], resp_estimate[n, i])]
    time_to_09_sig = np.argmin(np.abs(sig_estimate[..., change_timepoint:] - (c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX)),
                               -1)
    mean_time_to_09_sig = np.nanmean(time_to_09_sig, -1)
    std_time_to_09_sig = np.nanstd(time_to_09_sig, -1)
    fig = pl.shaded_errplot(n_space, mean_time_to_09_sig, std_time_to_09_sig,
                            plot_kwargs={'marker': 'o'},
                            shade_kwargs={'color': c.NT_COLOR},
                            labels={"title": "RT as a function of Hill coefficient", "xlabel": "Hill coefficient",
                                    "ylabel": "Reaction time"})
    pl.savefig(fig, "Updating speed as a function of Hill coefficient", si=True, ignore=fig.get_axes())
    pl.close_all()

    return mean_time_to_09_sig, std_time_to_09_sig


def rosenberg_simulation(ax=None):
    # slower updating, but by 1 time-point
    np.random.seed(c.SEED)
    change_timepoint, signal = prepare_signal()
    asd_c = 0.75 + c.SR_NT_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))
    nt_c = 1. + c.SR_NT_SIGMA_KM * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))
    # get responses
    asd_resp = utils.ei_gain_func(c.ROSENBERG_SCALING * signal, 1, 0.5, asd_c)
    nt_resp = utils.ei_gain_func(c.ROSENBERG_SCALING * signal, 1, 0.5, nt_c)
    # extract variances per step per repeat
    asd_var = asd_resp.var(-1) / c.SR_N_NEURONS
    nt_var = nt_resp.var(-1) / c.SR_N_NEURONS
    # run Kalman filter on the measured responses
    # initialize estimates and confidences:
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
    dense_sig = np.linspace(0, 1, 5000)
    gain_func = utils.ei_gain_func
    asd_dense_resp = np.squeeze(gain_func(c.ROSENBERG_SCALING * dense_sig[None, :, None], 1, 0.5, asd_c).mean(-1))
    nt_dense_resp = np.squeeze(gain_func(c.ROSENBERG_SCALING * dense_sig[None, :, None], 1, 0.5, nt_c).mean(-1))
    for i in tqdm(range(c.SR_NUM_REPS)):
        asd_sig_estimate[i] = dense_sig[np.searchsorted(asd_dense_resp[i], asd_resp_estimate[i])]
        nt_sig_estimate[i] = dense_sig[np.searchsorted(nt_dense_resp[i], nt_resp_estimate[i])]
    kalman_fig = pl.plot_kalman_filter(signal, asd_sig_estimate, nt_sig_estimate, text_x=(16, 13), text_y=(110, 300),
                                       fig=None if ax is None else ax.get_figure(), axs=ax,
                                       labels=["Decreased", "Full"])
    if ax is None:
        pl.savefig(kalman_fig, "rosenberg slower updating in asd", shift_x=-0.15, shift_y=1.1, tight=False,
                   numbering_size=30, ignore=kalman_fig.get_axes(), si=True)

    print("Fitting tapping data...")
    fitting_kwargs = dict(prior_var=0.1, perturb_prob=1e-6, base_n=20, n_neurons=200, perceptual_noise=0.03,
                          fit_scale_factor=50)
    asd_data = TappingData("ASD.mat", **fitting_kwargs)
    nt_data = TappingData("NT.mat", **fitting_kwargs)
    asd_data.fit_to_group_dynamics_rosenberg()
    nt_data.fit_to_group_dynamics_rosenberg()
    print(
        rf"Fitted Inhibitions:\nASD = {np.nanmean(asd_data.rosenberg_fitted_nu):.2g}$\pm${np.nanstd(asd_data.rosenberg_fitted_nu) / len(asd_data):.2g}"
        rf",\nNT = {np.nanmean(nt_data.rosenberg_fitted_nu):.2g}$\pm${np.nanstd(nt_data.rosenberg_fitted_nu) / len(nt_data):.2g}"
    )
    print("Mann Whitney U test ASD < NT Inhibitions, results:")
    print(scipy.stats.mannwhitneyu(asd_data.rosenberg_fitted_nu[~np.isnan(asd_data.rosenberg_fitted_nu)].ravel(),
                                   nt_data.rosenberg_fitted_nu[~np.isnan(nt_data.rosenberg_fitted_nu)].ravel(),
                                   alternative="less"))
    fig = pl.plot_tapping_fit_rosenberg(asd_data, color=c.ASD_COLOR, title="ASD group fit")
    pl.savefig(fig, "Rosenberg ASD group fit", si=True, ignore=fig.get_axes(), tight=True)
    pl.plt.close(fig)
    fig = pl.plot_tapping_fit_rosenberg(nt_data, color=c.NT_COLOR, title="NT group fit")
    pl.savefig(fig, "Rosenberg NT group fit", si=True, ignore=fig.get_axes(), tight=False)
    pl.plt.close(fig)


# %%
if __name__ == '__main__':
    simulate_signal_change_tracking_update()
    pl.plt.close('all')
