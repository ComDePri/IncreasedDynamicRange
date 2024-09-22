import constants as c
import idr_utils as utils
import lca_simulations
import matplotlib.pyplot as plt
import numpy as np
import plotting as pl
import scipy.stats
import scipy.stats as st
from belief_updating_sim import simulate_signal_change_tracking_update
from binocular_rivalry import simulate_binocular_rivalry
from data_fitting import TappingData
from encoding_capacity_sim import simulate_encoding_capacity
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from population_response_sim import simulate_population_response
from separatrix_sim import simulate_separatrix
from signal_differences_sim import simulate_signal_differences
from tqdm import tqdm
from variance_over_signal_range_sim import variance_simulation, get_variance

utils.reload(c)
utils.reload(pl)
plt.style.use('comdepri.mplstyle')
_ = utils.get_effective_n_from_heterogeneity(0.1)
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

# %% plots different sampling
pl.plot_effect_of_sampling_on_dynamic_range()
# %% SNR calculation
np.random.seed(0)
asd_n = 7
nt_n = 16
asd_total_snr = []
nt_total_snr = []
clean_sig = np.arange(0, 1.025, 0.025)
noise_levels = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
for signal_noise in tqdm(noise_levels):
    asd_var, _, nt_var, signal = get_variance(False, signal_noise, clean_sig.size)
    asd_var = asd_var.mean(-1)
    nt_var = nt_var.mean(-1)
    asd_real_diff = utils.hill_func(clean_sig, asd_n, 0.5)
    nt_real_diff = utils.hill_func(clean_sig, nt_n, 0.5)
    asd_real_diff = asd_real_diff[1:] - asd_real_diff[:-1]
    nt_real_diff = nt_real_diff[1:] - nt_real_diff[:-1]
    asd_snr = asd_real_diff / np.sqrt(asd_var[1:] + asd_var[:1])
    nt_snr = nt_real_diff / np.sqrt(nt_var[1:] + nt_var[:1])

    asd_total = scipy.integrate.simpson(y=asd_snr, x=clean_sig[1:])
    nt_total = scipy.integrate.simpson(y=nt_snr, x=clean_sig[1:])
    plt.figure(figsize=pl.get_fig_size(1, 1.5))
    plt.plot(clean_sig[1:], nt_snr, label=f"NDR")
    plt.plot(clean_sig[1:], asd_snr, label=f"IDR")
    plt.text(0.01, 0.78,
             rf"NDR $\int \Delta/Var(\Delta)$: {nt_total:.3f}\nIDR $\int \Delta/Var(\Delta)$: {asd_total:.2f}",
             fontsize=12, transform=plt.gca().transAxes)
    plt.legend()
    plt.title(f"Discriminability as a function of signal with noise {signal_noise:.3f}")
    plt.xlabel("Signal level")
    plt.ylabel(r"Discriminability ($\frac{A(S_n)-A(S_{n-1})}{\sqrt{Var(A(S_n))+Var(A(S_{n-1}))}}$)")
    pl.savefig(plt.gcf(), f"NDR and IDR SNR as a function of signal with noise {signal_noise:.3f}", tight=False,
               si=True)
    pl.close_all()
    asd_total_snr.append(asd_total)
    nt_total_snr.append(nt_total)

plt.figure(figsize=pl.get_fig_size(1, 1.3))
plt.plot(noise_levels, nt_total_snr, label="NDR", marker='o')
plt.plot(noise_levels, asd_total_snr, label="IDR", marker='o')
plt.legend()
plt.xlabel("Signal noise")
plt.ylabel(r"Total discriminability")
pl.savefig(plt.gcf(), "Total discriminability as a function of signal noise", tight=False, si=True)
# %%
np.random.seed(0)
nt_n = 16
asd_sd = 0.175
nt_sd = 0.01
N_REPS = 1000
N_NEURONS = 200
clean_sig = np.arange(0, 1.025, 0.025)
noise_levels = np.array([0.05, 0.1, 0.3, 0.4])
noise_level = 0.05
np.random.seed(c.SEED)
noisy_sig = clean_sig[:, None, None] + np.random.normal(0, noise_level, size=(clean_sig.shape + (N_REPS, N_NEURONS)))
asd_resp = utils.hill_func(noisy_sig, nt_n, 0.5 + np.random.uniform(-asd_sd, asd_sd, size=(1, N_REPS, N_NEURONS)))
nt_resp = utils.hill_func(noisy_sig, nt_n, 0.5 + np.random.uniform(-nt_sd, nt_sd, size=(1, N_REPS, N_NEURONS)))
asd_pop_diff = np.diff(asd_resp.mean(-1), axis=0)
nt_pop_diff = np.diff(nt_resp.mean(-1), axis=0)
asd_ci = np.percentile(asd_pop_diff, [25, 75], axis=-1)
nt_ci = np.percentile(nt_pop_diff, [25, 75], axis=-1)
fig = pl.shaded_errplot(clean_sig[1:], asd_pop_diff.mean(-1), asd_ci, plot_kwargs=dict(label="IDR", color=c.ASD_COLOR))
pl.shaded_errplot(clean_sig[1:], nt_pop_diff.mean(-1), nt_ci, plot_kwargs=dict(label="NDR", color=c.NT_COLOR),
                  ax=fig.get_axes()[0])
fig.get_axes()[0].set(xlabel="Signal level", ylabel="Population response difference",
                      title="Population response difference")
pl.savefig(fig, "Population response difference as a function of signal with noise", tight=False, si=True)
pl.close_all()
# %% combine power analysis
for ns in [(11, 9), (12, 8), (12, 10), (13, 7), (13, 9), (14, 6), (14, 8), (15, 7), (16, 8)]:
    data, individual_ns = utils.load_power_analysis_data(ns[0], ns[1])
    population, powers, heterogeneities = utils.parse_power_analysis_data(data, individual_ns)
    ax = pl.plot_combined_power_analysis(heterogeneities, ns[0], ns[1], population, powers)
    pl.savefig(ax.get_figure(), f"Combined power analysis for NDR={ns[0]} and IDR={ns[1]}", tight=False, si=True)
pl.close_all()


# %% power analysis for dynamic range
def get_width_and_ratio(signal, slope_resp):
    min_sig = signal[np.argmax(slope_resp >= 0.1 * slope_resp.max(0), axis=0)]
    max_sig = signal[np.argmax(slope_resp >= 0.9 * slope_resp.max(0), axis=0)]
    widths = max_sig - min_sig
    ratios = max_sig / min_sig
    return widths, ratios


# %% effect size analysis for dynamic range
signal = np.linspace(0, 1, c.PR_NUM_S_LEVELS)
slopes = np.linspace(3.2, 16, 50)
if utils.HETEROGENEITY_TO_N_STD is None:
    _ = utils.get_effective_n_from_heterogeneity(0.01)
slopes_std = utils.HETEROGENEITY_TO_N_STD[1][::-1][np.searchsorted(utils.HETEROGENEITY_TO_N[1][::-1], slopes)]
slope_resp = utils.hill_func(signal[:, None], slopes[None, :], c.PR_KM)
slope_resp_low_std = utils.hill_func(signal[:, None], slopes[None, :] - slopes_std[None, :], c.PR_KM)
slope_resp_high_std = utils.hill_func(signal[:, None], slopes[None, :] + slopes_std[None, :], c.PR_KM)
widths, ratios = get_width_and_ratio(signal, slope_resp)
low_std_widths, low_std_ratios = get_width_and_ratio(signal, slope_resp_low_std)
high_std_widths, high_std_ratios = get_width_and_ratio(signal, slope_resp_high_std)
slopes_fig = pl.plot_dynamic_range_as_function_of_slope(slopes, [widths, low_std_widths, high_std_widths],
                                                        [ratios, low_std_ratios, high_std_ratios])
pl.savefig(slopes_fig, "Dynamic range as a function of slope", numbering_size=25, si=True, tight=True, shift_y=1.05)
pl.close_all()
# %% neural variability - test the effect of different levels of signal noise on variance pattern
signal_sd_list = np.round([0.07, 0.13, 0.19], 4)  # levels of signal noise
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
    ax.plot(sig[:, 0], nt_variance.mean(1), alpha=0.8, c=c.NT_COLOR, label="NDR")
    scatt_asd = ax.scatter(sig, asd_variance, s=1, alpha=0.4, c=c.ASD_COLOR)
    ax.plot(sig[:, 0], asd_variance.mean(1), alpha=0.8, c=c.ASD_COLOR, label="IDR")
    scatt_ei = ax.scatter(sig, ei_variance, s=1, alpha=0.4, c=c.EI_COLOR)
    ax.plot(sig[:, 0], ei_variance.mean(1), alpha=0.8, c=c.EI_COLOR, label="E/I")
    lgnd = ax.legend([scatt_nt, scatt_asd, scatt_ei], ["NDR", "IDR", "E/I"], fontsize=25)
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
n_reps = 10
time_to_09_sig = np.zeros((noise_levels.size, n_reps)).astype(np.float32)
effective_ns = np.zeros((noise_levels.size, n_reps)).astype(np.float32)
for i, noise in enumerate(tqdm(noise_levels, desc="Noise levels")):
    for k in range(n_reps):
        np.random.seed(c.SEED)
        # 0.2-0.8 instead of 0-1 so the fit is around the Km, which is the important part for the slope
        effective_n_sig = np.linspace(0.2, 0.8, 150)
        km = 0.5 + noise * np.random.uniform(-1, 1, size=(1, 1, c.SR_N_NEURONS))
        effective_n_population_resp = utils.hill_func(effective_n_sig[:, None], c.SR_N, np.squeeze(km)[None, :])
        effective_ns[i, k] = utils.get_effective_n(effective_n_population_resp.mean(1), effective_n_sig, 0.5)
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
        time_to_09_sig[i, k] = np.mean(tmp[tmp > 0])
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

    axs.plot(asd_sig_estimate[c.SR_PLOT_REP_IDX, :], label="IDR", color=c.ASD_COLOR, linewidth=3)
    axs.plot(nt_sig_estimate[c.SR_PLOT_REP_IDX, :], label="NDR", color=c.NT_COLOR, linewidth=2, alpha=0.7)
    axs.plot(signal[c.SR_PLOT_REP_IDX, :, 0], color='gray', label="measured signal", alpha=0.3)
    axs.plot([c.SR_SIG_MIN] * change_timepoint +
             [c.SR_SIG_MAX] * (c.SI_SR_NUM_STEPS - change_timepoint), color='k',
             label="real signal", linewidth=3, zorder=1)

    subax = pl.inset_axes(axs, "50%", "50%", loc="lower right", borderpad=3, bbox_transform=axs.transAxes,
                          bbox_to_anchor=(0.05, 0.05, 1, 1))  # type: plt.Axes
    subax.hist(asd_times, bins=get_bins(asd_times, 50), label="IDR", density=True, color=c.ASD_COLOR)
    subax.hist(nt_times, bins=get_bins(nt_times, 50), label="NDR", density=True, color=c.NT_COLOR)
    subax.text(text_x[1], text_y[1], "IDR", color=c.ASD_COLOR, size=25, weight="bold")
    subax.text(text_x[0], text_y[0], "NDR", color=c.NT_COLOR, size=25, weight="bold")

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
    [3.3, 15], [3.3, 35],
    [7, 75], [15, 190]
]
# ASD,NT
text_y_list = [
    [0.75, 0.4], [0.75, 0.25],
    [0.6, 0.2], [0.6, 0.2]
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
ax.axhline(nt_times, linestyle='--', color=c.NT_COLOR, label="NDR")
ax.axhline(asd_times, linestyle='--', color=c.ASD_COLOR, label="IDR")
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
encoding_range_space = np.linspace(0.2, 0.35, 7)
for encoding_range in tqdm(encoding_range_space, desc="Encoding range"):
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
ax.plot(encoding_range_space, asd_mean_n, marker='o', label="IDR", color=c.ASD_COLOR)
ax.plot(encoding_range_space, nt_mean_n, marker='o', label="NDR", color=c.NT_COLOR)
ax.plot(encoding_range_space, ei_mean_n, marker='o', label="E/I", color=c.EI_COLOR)
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
                 group_labels=["NDR", "IDR"],
                 title="pure-mixed ratio over threshold",
                 xlabel="Pure state threshold", ylabel="Ratio $\\frac{Pure_t}{Mixed_t}$",
                 colors=[c.NT_COLOR, c.ASD_COLOR],
                 p_val_test_func=lambda x, y: st.ttest_rel(x, y)[1])

# %% effect of noise on population level slope
if utils.HETEROGENEITY_TO_N is None:
    _ = utils.get_effective_n_from_heterogeneity(0.1)
noise_level_list, retrieved_n = utils.HETEROGENEITY_TO_N
_, retrieved_n_std = utils.HETEROGENEITY_TO_N_STD
# plot
fig = plt.figure(figsize=pl.get_fig_size(1, 1.7))
ax: plt.Axes = fig.subplots()
ax.plot(noise_level_list, retrieved_n, color=c.ASD_COLOR)
ax.fill_between(noise_level_list, retrieved_n - retrieved_n_std, retrieved_n + retrieved_n_std, color=c.ASD_COLOR,
                alpha=0.3)
pl.set_ax_labels(ax, "Increased heterogeneity in half activation point\nreduces slope (n) of population response",
                 "Heterogeneity level", "Hill coefficient (n)")
subax = inset_axes(ax, "40%", "30%", loc="upper right", borderpad=2)
n_deriv = np.gradient(retrieved_n, noise_level_list[1] - noise_level_list[0])
box = np.ones(20) / 20
y_smooth = np.convolve(n_deriv, box, mode='same')
subax.plot(noise_level_list, y_smooth, color=c.ASD_COLOR)
subax.set(xlabel="Heterogeneity level", ylabel=r"$\frac{dn}{d\sigma}$")
pl.savefig(fig, "effect of noise on population slope", ignore=fig.get_axes(), si=True, shift_x=-0.09, shift_y=1.05)
plt.close(fig)

# %% generate encoding figure
x = np.linspace(0, 1, 1000)
resp1 = utils.ei_gain_func(x, 16, 0.5)
resp2 = utils.ei_gain_func(x, 4, 0.5)
fig, ax = plt.subplots(figsize=pl.get_fig_size(1, 1))
ax.plot(x, resp1, color=c.NT_COLOR, linewidth=15)
ax.plot(x, resp2, color=c.ASD_COLOR, linewidth=15)
ax.set_ylabel("$f(I)$", fontsize=60)
ax.set_xlabel("$I$", fontsize=60)
fig.savefig("data/Encoding figure.pdf")
pl.close_all()
# %% divisive normalization and IDR
s1 = np.linspace(0, 1, 1001)
s2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 1])

ndr_resp1 = utils.hill_func(s1, 16, 0.5)
ndr_resp2 = utils.hill_func(s2, 16, 0.5)
idr_resp1 = utils.hill_func(s1, 8, 0.5)
idr_resp2 = utils.hill_func(s2, 8, 0.5)

ndr_div_norm = ndr_resp1[:, None] / (ndr_resp1[:, None] + ndr_resp2[None, :])
idr_div_norm = idr_resp1[:, None] / (idr_resp1[:, None] + idr_resp2[None, :])

fig, ax = plt.subplots(1, 1, figsize=pl.get_fig_size(0.7, 1))
ax.plot(s1, ndr_div_norm)  # , label=[f"NDR /{s:.1f}" for s in s2])
ax.set_prop_cycle(None)
ax.plot(s1, idr_div_norm, linestyle=":", alpha=0.7)  # , label=[f"IDR /{s:.1f}" for s in s2]
ax.set_prop_cycle(None)
ax.set_xticks(s2, [f"{s:.2g}" for s in s2], fontsize=10)

ax.set_title("IDR and divisive normalization", fontsize=18)
pl.savefig(fig, "Divisive normalization and IDR", si=True, numbering_size=30, shift_x=-0.1, shift_y=1.05)
pl.close_all()

# %% divisive normalization for rosenberg
s1 = c.ROSENBERG_SCALING * np.linspace(0, 1, 1001)
s2 = c.ROSENBERG_SCALING * np.array([0.1, 0.2, 0.3, 0.8])

nt_resp1 = utils.ei_gain_func(s1, 1, 0.5, 1)
nt_resp2 = utils.ei_gain_func(s2, 1, 0.5, 1)
asd_resp1 = utils.ei_gain_func(s1, 1, 0.5, 0.75)
asd_resp2 = utils.ei_gain_func(s2, 1, 0.5, 0.75)

nt_div_norm = nt_resp1[:, None] / (nt_resp1[:, None] + nt_resp2[None, :])
asd_div_norm = asd_resp1[:, None] / (asd_resp1[:, None] + asd_resp2[None, :])

fig, ax = plt.subplots(figsize=pl.get_fig_size(.7, 1.2))
ax.plot(s1, nt_div_norm)  # , label=[f"NDR /{s:.1f}" for s in s2])
ax.set_prop_cycle(None)
ax.plot(s1, asd_div_norm, linestyle=":", alpha=0.7)  # , label=[f"IDR /{s:.1f}" for s in s2]
# ax.vlines(s2,0,1,colors='k',linestyles="--",linewidths=1,alpha=0.5)
# ax.grid(False)
# ax.legend()
ax.set_xticks(s2, [f"{s:.2g}" for s in s2], fontsize=10)
ax.set_title("Divisive normalization and decreased inhibition")
pl.savefig(fig, "Rosenberg divisive normalization", si=True, ignore=[ax])
pl.close_all()
# %%
fig, ax = plt.subplots(figsize=pl.get_fig_size(0.8, 1.4))
sig = np.arange(0.1, 0.91, 0.01)
nt_resp = utils.hill_func(sig, 16, 0.5)
asd_resp = utils.hill_func(sig, 7, 0.5)
ax.plot(sig[1:], np.diff(asd_resp) / np.diff(nt_resp), label="Ratio")
ax.hlines(1, 0, 1, 'k', '--', linewidths=1.5, alpha=0.6, label="Equality")
ax.semilogy(base=10)
ax.set(xlabel="Signal", ylabel=r"Signal difference ratio $\frac{\Delta IDR}{\Delta NDR}$",
       title="Discriminability ratio")
ax.legend()
pl.savefig(fig, "Discriminability ratio", si=True, ignore=[ax])

# %% simulate different parameters for LCA.
params = [
    dict(threshold=0.51, n_sim=200, leak=1, inhibition=0.25, noise=0.32),
    dict(threshold=0.65, n_sim=200, leak=1, inhibition=0.25, noise=0.32),
    dict(threshold=0.51, n_sim=200, leak=1, inhibition=0.1, noise=0.32),
    dict(threshold=0.51, n_sim=200, leak=1, inhibition=0.25, noise=0.4)
]
fig1 = plt.figure(figsize=pl.get_fig_size(2, 2.5))
fig2 = plt.figure(figsize=pl.get_fig_size(2, 2.5))
gs1 = fig1.add_gridspec(2, 2)
ax1 = fig1.add_subplot(gs1[0, 0])
axes1 = [ax1, fig1.add_subplot(gs1[0, 1], sharey=ax1), fig1.add_subplot(gs1[1, 0], sharey=ax1),
         fig1.add_subplot(gs1[1, 1], sharey=ax1)]
gs2 = fig2.add_gridspec(2, 2)
axes2 = [fig2.add_subplot(gs2[0, 0]), fig2.add_subplot(gs2[0, 1]), fig2.add_subplot(gs2[1, 0]),
         fig2.add_subplot(gs2[1, 1])]
for p, ax_idr, ax_ei in zip(params, axes1, axes2):
    asd_correct_percent, ei_correct_percent, nt_correct_percent, signal_levels, times = lca_simulations.lca_robertson_simulation(
        **p)
    nt_threshold_idx = np.argmax(nt_correct_percent > 0.8, axis=1)
    nt_threshold_idx[nt_correct_percent.max(axis=1) < 0.8] = signal_levels.size - 1
    asd_threshold_idx = np.argmax(asd_correct_percent > 0.8, axis=1)
    asd_threshold_idx[asd_correct_percent.max(axis=1) < 0.8] = signal_levels.size - 1
    ei_threshold_idx = np.argmax(ei_correct_percent > 0.8, axis=1)
    ei_threshold_idx[ei_correct_percent.max(axis=1) < 0.8] = signal_levels.size - 1

    pl.plot_robertson_vs_lca(ax_ei, None, nt_threshold_idx, ei_threshold_idx, signal_levels, times, 0.158,
                             c.EI_COLOR, labels=["Full inhibition", "Decreased inhibition"])
    ax_ei.set_title(f"Threshold={p['threshold']:.2g}, Inhibition={p['inhibition']:.2g}, Noise={p['noise']:.2g}")
    pl.plot_robertson_vs_lca(ax_idr, None, nt_threshold_idx, asd_threshold_idx, signal_levels, times, 0.158,
                             labels=["NDR", "IDR"])
    ax_ei.set_title(f"Threshold={p['threshold']:.2g}, Inhibition={p['inhibition']:.2g}, Noise={p['noise']:.2g}")
    ax_idr.set_title(f"Threshold={p['threshold']:.2g}, Inhibition={p['inhibition']:.2g}, Noise={p['noise']:.2g}")
pl.savefig(fig2, "LCA with different parameters decreased inhibition", si=True, numbering_size=30, shift_x=-0.1,
           shift_y=1.075)
pl.savefig(fig1, "LCA with different parameters", si=True, shift_x=-0.1,
           shift_y=1.075)
pl.close_all()

# %% simulate rosenberg LCA with different parameters.
params = [
    dict(threshold=0.6, n_sim=200, leak=.51, inhibition=0.2, noise=0.3),
    dict(threshold=0.55, n_sim=200, leak=.51, inhibition=0.2, noise=0.3),
    dict(threshold=0.6, n_sim=200, leak=.51, inhibition=0.1, noise=0.3),
    dict(threshold=0.6, n_sim=200, leak=.51, inhibition=0.2, noise=0.4)
]
fig = plt.figure(figsize=pl.get_fig_size(2, 2.5))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
axes = [ax1, fig.add_subplot(gs[0, 1], sharey=ax1), fig.add_subplot(gs[1, 0], sharey=ax1),
        fig.add_subplot(gs[1, 1], sharey=ax1)]
for p, ax in zip(params, axes):
    asd_correct_percent, nt_correct_percent, signal_levels, times = lca_simulations.simulate_rosenberg(
        ax1=ax, title=f"Threshold={p['threshold']:.2g}, Inhibition={p['inhibition']:.2g}, Noise={p['noise']:.2g}", **p)
pl.savefig(fig, "Rosenberg LCA with different parameters", shift_x=-0.1, shift_y=1.075,si=True)
pl.close_all()

# %% plot HGF
asd_fit, sim_fit, alphas, omegas = utils.load_hgf()
fig = pl.plot_hgf(asd_fit, sim_fit, alphas, omegas)
pl.savefig(fig, "hgf model", si=True, tight=True, shift_x=-0.1, shift_y=1.1, numbering_size=30)
pl.close_all()
