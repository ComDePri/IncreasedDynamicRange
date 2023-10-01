import numpy as np
import matplotlib.pyplot as plt
import constants as c
import plotting as pl
from scipy.optimize import curve_fit
import scipy.stats as st
import scipy.stats
import idr_utils as utils
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
from matplotlib.gridspec import GridSpec
from belief_updating_sim import simulate_signal_change_tracking_update
from binocular_rivalry import simulate_binocular_rivalry
from encoding_capacity_sim import simulate_encoding_capacity
from population_response_sim import simulate_population_response
from separatrix_sim import simulate_separatrix
from signal_differences_sim import simulate_signal_differences
from variance_over_signal_range_sim import variance_simulation
from data_fitting import TappingData

utils.reload(c)
utils.reload(pl)
import itertools as it

plt.style.use('comdepri.mplstyle')
# %% Logistic func simulations - main simulations with logistic function instead of Hill function
print("=====================================\n"
      "========= Logistic function ========\n"
      "=====================================")
simulate_signal_differences(si=True)  # Sensitivity to signal differences

variance_simulation(si=True)  # Variance over signal range - E/I VS IDR

simulate_signal_change_tracking_update(si=True)  # Slower responses to sharp transitions using kalman filter

simulate_binocular_rivalry(si=True)  # binocluar rivalry - noisy signal around 0.5

simulate_encoding_capacity(si=True)  # FI based encoding capacity

simulate_population_response(si=True)  # Population response

simulate_separatrix(si=True)  # separatrix

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
fig = plt.figure(figsize=pl.get_fig_size(1.2, 2.2))
gs = GridSpec(2, 4, left=0.08, hspace=0.3, wspace=0.2)
axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:]), fig.add_subplot(gs[1, 1:3])]
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
    ax.set_title(r"$\sigma^2_{signal}=%.2f$" % signal_sd)
    scatt_nt = ax.scatter(sig, nt_variance, s=1, alpha=0.4, c=c.NT_COLOR)
    ax.plot(sig[:, 0], nt_variance.mean(1), alpha=0.8, c=c.NT_COLOR, label="NT")
    scatt_asd = ax.scatter(sig, asd_variance, s=1, alpha=0.4, c=c.ASD_COLOR)
    ax.plot(sig[:, 0], asd_variance.mean(1), alpha=0.8, c=c.ASD_COLOR, label="ASD")
    scatt_ei = ax.scatter(sig, ei_variance, s=1, alpha=0.4, c=c.EI_COLOR)
    ax.plot(sig[:, 0], ei_variance.mean(1), alpha=0.8, c=c.EI_COLOR, label="E/I")
    lgnd = ax.legend([scatt_nt, scatt_asd, scatt_ei], ["NT", "ASD", "E/I"], fontsize=25)
    for handle in lgnd.legendHandles:
        handle.set_sizes([200.0])
    max_var = max(max_var, nt_variance.max(), asd_variance.max(), ei_variance.max())
for ax in axes:
    ax.set_ylim(0, max_var * 1.1)

fig.text(0.02, 0.5, "Variance in population responses", fontsize=27, ha='center', va='center', rotation=90,
         fontweight='bold')
pl.savefig(fig, "Variance pattern with different levels of signal noise", tight=False, si=True,
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

# %%
inhibitions = np.linspace(0.1, 1, 10)
mean_rts = []
std_rts = []


def tracking(n, nu, km_sigma):
    signal = np.zeros((c.SR_NUM_REPS, c.SR_NUM_STEPS, c.SR_N_NEURONS))
    change_timepoint = round(max(c.SR_NUM_STEPS * c.SR_MIN_SIG_PERCENTAGE, c.SR_MIN_SIG_TIMEPOINT))
    signal[:, :change_timepoint, :] = c.SR_SIG_MIN
    signal[:, change_timepoint:, :] = c.SR_SIG_MAX
    signal += np.random.normal(scale=c.SR_SIG_SIGMA, size=signal.shape)
    signal[signal < 0] = 0
    signal[signal > 1] = 1
    km = 0.5 + km_sigma * np.random.uniform(-1, 1, size=(c.SR_NUM_REPS, 1, c.SR_N_NEURONS))
    resp = utils.ei_gain_func(signal, n, km, nu)
    var = resp.var(-1) / c.SR_N_NEURONS
    estimated_var = np.zeros_like(var, dtype=np.float64)
    estimated_var[:, 0] = c.SR_PRIOR_VARIANCE
    resp_estimate = np.zeros_like(var, dtype=np.float64)
    kg = np.zeros_like(var, dtype=np.float64)
    resp = resp.mean(-1)
    for j in range(1, c.SR_NUM_STEPS):
        utils.kalman_filter_step(j, kg, estimated_var, var, resp_estimate, resp)
    sig_estimate = np.zeros_like(resp_estimate)
    dense_sig = np.linspace(0, 1, 10000)
    dense_resp = np.squeeze(utils.ei_gain_func(dense_sig[None, :, None], n, km, nu).mean(-1))
    for j in range(c.SR_NUM_REPS):
        sig_estimate[j] = dense_sig[np.searchsorted(dense_resp[j], resp_estimate[j])]
    times = np.argmax(np.abs(sig_estimate) >= c.SR_TRACK_PERCENTAGE * c.SR_SIG_MAX, axis=1) - change_timepoint
    return times


np.random.seed(c.SEED)
for inhibition in tqdm(inhibitions, desc="Inhibition strength"):
    ei_times = tracking(c.SR_N, inhibition, c.SR_NT_SIGMA_KM)
    mean_rts.append(np.mean(ei_times[ei_times > 0]))
    std_rts.append(np.std(ei_times[ei_times > 0]))

np.random.seed(c.SEED)
asd_times = tracking(c.SR_N, 1, c.SR_ASD_SIGMA_KM).mean()
nt_times = tracking(c.SR_N, 1, c.SR_NT_SIGMA_KM).mean()
# %%
fig = plt.figure()
ax: plt.Axes = fig.subplots()
ax.axhline(nt_times, linestyle='--', color=c.NT_COLOR, label="NT")
ax.axhline(asd_times, linestyle='--', color=c.ASD_COLOR, label="ASD")
ax.errorbar(inhibitions, mean_rts, std_rts, marker='o', color=c.EI_COLOR, label="E/I")
ax.set_xlabel("Inhibition strength")
ax.set_ylabel("Mean reaction time")
ax.legend()
pl.savefig(fig, "Reaction time as a function of inhibition strength", ignore=[ax], si=True)
plt.close()

# %% effect of encoding on asd/nt/ei fits
fitting_kwargs = dict(prior_var=0.1, perturb_prob=1e-6, base_n=20, n_neurons=200, perceptual_noise=0.03,
                      fit_scale_factor=50)
asd_mean_n = []
nt_mean_n = []
ei_mean_n = []
for encoding_range in tqdm(np.linspace(0.15, 0.4, 11), desc="Encoding range"):
    asd_data = TappingData("ASD.mat", half_range=encoding_range, **fitting_kwargs)
    nt_data = TappingData("NT.mat", half_range=encoding_range, **fitting_kwargs)
    ei_data = TappingData("ASD.mat", half_range=encoding_range, **fitting_kwargs)
    asd_data.fit_to_group_dynamics()
    nt_data.fit_to_group_dynamics()
    ei_data.fit_ei_to_group_dynamics(min_n=np.nanmean(nt_data.fitted_n), nu_range=[0.1, 1.])
    asd_mean_n.append(np.nanmean(asd_data.fitted_n))
    nt_mean_n.append(np.nanmean(nt_data.fitted_n))
    ei_mean_n.append(np.nanmean(ei_data.fitted_n))

fig = plt.figure()
ax: plt.Axes = fig.subplots()
ax.plot(np.linspace(0.15, 0.4, 11), asd_mean_n, marker='o', label="ASD", color=c.ASD_COLOR)
ax.plot(np.linspace(0.15, 0.4, 11), nt_mean_n, marker='o', label="NT", color=c.NT_COLOR)
ax.plot(np.linspace(0.15, 0.4, 11), ei_mean_n, marker='o', label="E/I", color=c.EI_COLOR)
ax.set_xlabel("Encoding range")
ax.set_ylabel("Mean Hill-coefficient")
ax.legend()
pl.savefig(fig, "Mean fitted Hill-coefficient as a function of encoding range", ignore=[ax], si=True)
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
    asd_transition_counts[:, i], asd_ratios[:, i], \
        _, _ = utils.get_binocular_rivalry_stats(asd_resp, low_thresh=low_thresholds[i],
                                                 high_thresh=high_thresholds[i])
    nt_transition_counts[:, i], nt_ratios[:, i], \
        _, _ = utils.get_binocular_rivalry_stats(nt_resp, low_thresh=low_thresholds[i],
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
omegas = hgf_mat["omegas"]


def tapas_sgm(x, a):
    return a / (1 + np.exp(-x))


def plot_param_trajectory(r, ax, title):
    t = np.ones_like(r.u)
    ts = np.concatenate([[0], np.cumsum(t)])
    ax.plot(ts, np.concatenate([[tapas_sgm(r.p_prc.mu_0[1], 1)], tapas_sgm(r.traj.mu[:, 1], 1)]),
            color='r', linewidth=4, label=r"$P(x_1=1)$")
    ax.scatter(0, tapas_sgm(r.p_prc.mu_0[1], 1), color="red", s=15)
    ax.plot(ts[1:np.argmax(r.u == 1) + 1], r.u[:np.argmax(r.u == 1)], color=[0, 0.6, 0], label="$u$, Input",
            linewidth=3,
            linestyle=":")
    ax.plot(ts[np.argmax(r.u == 1) + 1:], r.u[np.argmax(r.u == 1):], color=[0, 0.6, 0], linewidth=3, linestyle=":")
    ax.plot(ts[1:np.argmax(r.y == 1) + 1], (((r.y - 0.5) * 1.05) + 0.5)[:np.argmax(r.y == 1)], color=[1, 0.7, 0],
            label="$y$, Response", linewidth=3, linestyle=":")
    ax.plot(ts[np.argmax(r.y == 1) + 1:], (((r.y - 0.5) * 1.05) + 0.5)[np.argmax(r.y == 1):], color=[1, 0.7, 0],
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
    ax.set_xticklabels(["ASD", "NT"], fontweight='bold')
    ax.tick_params(axis='y', which='major', labelsize=17)
    ax.tick_params(axis='y', which='minor', labelsize=17)


fig, axes = plt.subplots(2, 2, figsize=pl.get_fig_size(1, 1.6))
axes = np.ravel(axes)

plot_param_trajectory(sim_fit, axes[0], "Base parameters fit")
plot_param_trajectory(asd_fit, axes[1], "High PU parameters fit")
plot_boxplots(alphas, axes[2], "Perceptual uncertainty", r"$\alpha$")
axes[2].set_ylim([0, axes[2].get_ylim()[1]])
plot_boxplots(omegas, axes[3], "Volatility estimates", r"$\omega_3$")

pl.savefig(fig, "hgf model", si=True, tight=True, shift_x=-0.15, shift_y=1.05, numbering_size=30)
