import numpy as np
import constants as c
import idr_utils as utils
import plotting as pl

pl.plt.style.use('comdepri.mplstyle')


def variance_simulation(si=False):
    print("======================================\n"
          "========= Variance Simulation ========\n"
          "======================================")
    np.random.seed(c.SEED)
    noisy_signal, signal = generate_signal()
    print("generating populations...")
    asd_km, nt_km = generate_km()
    print("calculating gain...")
    asd_gain, ei_gain, nt_gain = calculate_population_responses(asd_km, noisy_signal, nt_km, si)
    del nt_km, asd_km, noisy_signal
    print("calculating variance...")
    asd_variance, ei_variance, nt_variance = calculate_variance_patterns(asd_gain, ei_gain, nt_gain)
    # calculate CI of .5 and 1/e from max variance
    width_ci_high, width_ci_low, widths = get_bootstrapped_widths(asd_variance, ei_variance, nt_variance)
    print(
        f"ASD variance width (max/2): {widths[0][0]:.4g}, 99% CI: [{width_ci_low[0][0]:.4g}, {width_ci_high[0][0]:.4g}]\n"
        f"E\I variance width (max/2): {widths[0][1]:.4g}, 99% CI: [{width_ci_low[0][1]:.4g}, {width_ci_high[0][1]:.4g}]\n"
        f"NT variance width (max/2): {widths[0][2]:.4g}, 99% CI: [{width_ci_low[0][2]:.4g}, {width_ci_high[0][2]:.4g}]\n"
        f"ASD variance width (max/e): {widths[1][0]:.4g}, 99% CI: [{width_ci_low[1][0]:.4g}, {width_ci_high[1][0]:.4g}]\n"
        f"E\I variance width (max/e): {widths[1][1]:.4g}, 99% CI: [{width_ci_low[1][1]:.4g}, {width_ci_high[1][1]:.4g}]\n"
        f"NT variance width (max/e): {widths[1][2]:.4g}, 99% CI: [{width_ci_low[1][2]:.4g}, {width_ci_high[1][2]:.4g}]\n")
    savename_prefix = "logistic_func/" if si else ""
    var_fig = pl.plot_variance_over_signal_range(signal[:, 0, 0, :], nt_variance, asd_variance, ei_variance)
    pl.savefig(var_fig, savename_prefix + "variance over signal range", tight=False, ignore=var_fig.get_axes(), si=si)


def get_bootstrapped_widths(asd_variance, ei_variance, nt_variance):
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
    return width_ci_high, width_ci_low, widths


def calculate_variance_patterns(asd_gain, ei_gain, nt_gain):
    nt_variance = nt_gain.mean(1).var(1).astype(np.float32)
    del nt_gain
    asd_variance = asd_gain.mean(1).var(1).astype(np.float32)
    del asd_gain
    ei_variance = ei_gain.mean(1).var(1).astype(np.float32)
    del ei_gain
    return asd_variance, ei_variance, nt_variance


def calculate_population_responses(asd_km, noisy_signal, nt_km, si):
    if si:
        n = c.SI_NV_N
        gain_func = utils.logistic_func
    else:
        n = c.NV_N
        gain_func = utils.ei_gain_func
    nt_gain = gain_func(noisy_signal, n, nt_km).astype(np.float32)
    ei_gain = gain_func(noisy_signal, n, nt_km, c.NU).astype(np.float32)
    asd_gain = gain_func(noisy_signal, n, asd_km).astype(np.float32)
    return asd_gain, ei_gain, nt_gain


def generate_km():
    nt_km = c.NV_KM + c.NV_NT_KM_SD * np.random.uniform(-1, 1, size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
        np.float32)
    asd_km = c.NV_KM + c.NV_ASD_KM_SD * np.random.uniform(-1, 1, size=(1, c.NV_NUM_NEURONS, 1, c.NV_PR_REPEATS)).astype(
        np.float32)
    return asd_km, nt_km


def generate_signal():
    print("generating signal...")
    signal = np.repeat(np.linspace(0, 1, c.NV_NUM_S), c.NV_REPEATS * c.NV_NUM_NEURONS * c.NV_PR_REPEATS).reshape(
        (c.NV_NUM_S, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal = signal + np.random.uniform(-c.NV_SIGNAL_SD, c.NV_SIGNAL_SD, size=(
        c.NV_NUM_S, c.NV_NUM_NEURONS, c.NV_REPEATS, c.NV_PR_REPEATS)).astype(np.float32)
    noisy_signal[noisy_signal < 0] = 0
    noisy_signal[noisy_signal > 1] = 1
    return noisy_signal, signal


if __name__ == '__main__':
    variance_simulation()
