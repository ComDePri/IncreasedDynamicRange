import importlib
import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.stats as st
from pypdf import PdfReader, PdfWriter, Transformation
from scipy.optimize import curve_fit
from scipy.stats import truncnorm as tn
from tqdm import tqdm

import constants as c
import plotting as pl
from data_fitting import TappingData


def reload(module):
    """ Reloads a given module """
    importlib.reload(module)


def rolling_window(a, window_size):
    """
    Returns a view of the given matrix that is a rolling window over the last axis
    :param a: Matrix
    :param window_size: size of window
    :return: new view of a with new axis that is the length of the window
    """
    shape = a.shape[:-1] + (a.shape[-1] - window_size + 1, window_size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_variance(a, window_size, axis=-1):
    """
    Calculates rolling window variance of the given matrix over the given axis
    :param a: Matrix
    :param window_size: size of the rolling window on which we calculate cariance
    :param axis: axis for variance calculation
    """
    return np.var(rolling_window(a, window_size), axis=axis)


def get_derivative_kernel(n):
    filt = np.ones(n + 1) / n
    filt[filt.size // 2] = 0
    filt[:filt.size // 2] *= -1
    return filt


def make_fig_dir():
    """
    Create a directory for figures if it doesn't already exist.
    Adds FIG_DIR variable to constants if it does not exist.
    """
    if "FIG_DIR" not in vars(c).keys():
        vars(c)["FIG_DIR"] = r"figures"
    path = os.path.join(os.curdir, c.FIG_DIR)
    os.makedirs(path, exist_ok=True)
    c.FIG_DIR = path


def make_si_fig_dir():
    """
    Create a directory for SI figures if it doesn't already exist.
    Adds SI_FIG_DIR variable to constants if it does not exist.
    """
    if "SI_FIG_DIR" not in vars(c).keys():
        vars(c)["SI_FIG_DIR"] = r"SI figures"
    path = os.path.join(os.curdir, c.SI_FIG_DIR)
    if not os.path.exists(path):
        os.mkdir(path)
    c.SI_FIG_DIR = path


def hill_func(s, n, km):
    """
    Hill function
    :param s: Signal level
    :param n: Slope
    :param km: Half activation point
    :return: the Hill function given the parameters and signal
    """
    sn = (s ** n)
    return sn / (sn + (km ** n))


def f_test(x, y, alternative="two_sided"):
    """
    Calculates the F-test.
    :param x: The first group of data
    :param y: The second group of data
    :param alternative: The alternative hypothesis, one of "two_sided" (default), "greater" or "less"
    :return: a tuple with the F statistic value and the p-value.
    """
    df1 = len(x) - 1
    df2 = len(y) - 1
    f = x.var() / y.var()
    if alternative == "greater":
        p = 1.0 - st.f.cdf(f, df1, df2)
    elif alternative == "less":
        p = st.f.cdf(f, df1, df2)
    else:
        # two-sided by default
        # Crawley, the R book, p.355
        p = 2.0 * (1.0 - st.f.cdf(f, df1, df2))
    return f, p


def population_resp_fi(s, n, km, nu=1.):
    """
    Fisher Information function of Hill function.
    :param s: Signal level
    :param n: Slope
    :param km: Half activation point
    :param nu: The inhibition factor. Default 1.
    :return:
    """
    return ((s ** (n - 2)) * (km ** (2 * n)) * (n ** 2)) / (nu * (s ** n) + (km ** n)) ** 3


def population_resp_fi_logistic(s, n, km, nu=1.):
    """
    Fisher Information function of logistic function.
    :param s: Signal level
    :param n: Slope
    :param km: Half activation point
    :return:
    """
    minus_n_times_diff = -n * (s - km)
    return ((n ** 2) * np.exp(2 * minus_n_times_diff)) / ((nu + np.exp(minus_n_times_diff)) ** 3)


def ei_gain_func(s, n, km, nu=1.):
    """
    Hill function with inhibition scaling
    :param s: Signal level
    :param n: Slope
    :param km: Half activation point
    :param nu: Inhibition scaling
    :return: the Hill function given the parameters and signal
    """
    return (s ** n) / (nu * (s ** n) + (km ** n))


def inverse_gain_func(activity, n, km):
    """
    The inverse of the Hill function. This is the optimal decoding of neuron activity to signal level.
    :param activity: Neuronal activation
    :param n: Neuronal slope
    :param km: Neuronal half-activation points
    :return: The signal decoded from the activity
    """
    return np.power((activity * (km ** n)) / (1 - activity), 1 / n)


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
    kg[..., t] = (last_est_var + delta) / (
            (last_est_var + delta) + meas_var[..., t])
    est_var[..., t] = (1 - kg[..., t]) * (last_est_var + delta)
    est_sig[..., t] = est_sig[..., t - 1] + kg[..., t] * (
            sig[..., t] - est_sig[..., t - 1])


def get_binocular_rivalry_stats(resp, low_thresh=c.BR_PURE_LOW_THRESH, high_thresh=c.BR_PURE_HIGH_THRESH):
    """
    Calculate the transition counts and pure state ratio from binocular rivalry simulation results
    :param resp: The simulation results
    :param low_thresh: threshold where resp<low_thresh means pure state A
    :param high_thresh: threshold where resp>high_thresh means pure state B
    :return: transition counts, pure-to-mixed state ratio, mixed state count
    """
    states_mat = np.zeros_like(resp)
    states_mat[resp < low_thresh] = -1
    states_mat[high_thresh < resp] = 1
    transitions_counts = np.zeros(c.BR_NUM_REPS)
    for j in range(c.BR_NUM_REPS):
        transitions_counts[j] = np.count_nonzero(np.diff(states_mat[j][np.nonzero(states_mat[j])]))
    pure_states_count = np.count_nonzero(states_mat, axis=1)
    mixed_state_count = (c.BR_NUM_STEPS - pure_states_count)
    ratio = pure_states_count / mixed_state_count
    return transitions_counts, ratio, mixed_state_count, (mixed_state_count != 0)


def get_ci(a: [np.array, list], stat_func: callable, alpha: float = 0.05, axis=-1, num_boot=10000,
           plot_params: dict = None):
    """
    calculates the confidence interval using bootstrap
    :param a: array of the observations
    :param stat_func: callable, The statistic function for which to calculate CI (for example np.mean, np.var).
    Must have axis parameter
    :param alpha: The confidence level. Default 0.05
    :param plot_params: dict. Supported for 1D arrays only.
                        If passed, plots the bootstrap distribution.
                        Should have 2 keys with string values - data_desc and func.
                            * data_desc - description of the data that is plotted.
                            * func - the name of the function that CI is calculated for.
    :return: the alpha,1-alpha quantiles of the resulting bootstrap distribution
    """
    CI = namedtuple('CI', ['low', 'high'])
    a = np.asarray(a)
    if axis < 0:
        axis = len(a.shape) + axis
    seed = 97
    np.random.seed(seed)
    if len(a.shape) == 1:  # 1D
        boot_samp = np.random.choice(a, (num_boot, a.size))
        boot_dist = stat_func(boot_samp, axis=1)
        low_ci, high_ci = np.quantile(boot_dist, [alpha / 2, 1 - alpha / 2])
    else:
        boot_idx = np.random.choice(np.arange(a.shape[axis]), (num_boot, a.shape[axis]))
        if axis + 1 < len(a.shape):
            boot_dist = np.zeros(a.shape[:axis] + a.shape[axis + 1:] + (num_boot,), dtype=np.float64)
        else:
            boot_dist = np.zeros(a.shape[:axis] + (num_boot,), dtype=np.float64)
        for rep in tqdm(range(num_boot), desc="Bootstrapping"):
            boot_samp = np.take(a, boot_idx[rep], axis=axis)
            boot_dist[..., rep] = stat_func(boot_samp, axis=axis)
        low_ci, high_ci = np.quantile(boot_dist, [alpha / 2, 1 - alpha / 2], axis=-1)
        return CI(low_ci, high_ci)

    if plot_params is not None:
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()
        ax.hist(boot_dist, 30, density=True, color=c.HIST_COLOR)
        ax.axvspan(low_ci, high_ci, color=c.FILL_COLOR, alpha=0.3)
        ax.set(title=f"Confidence Interval for {plot_params['data_desc']} {plot_params['func']}")
        fig.tight_layout()
    return CI(low_ci, high_ci)


def separatrics_model(y0, t, J, km, signal_sigma, nu):
    """
    Function for running separatrix in euler.
    :param y0: neuron activation
    :param t: time point
    :param J: connectivity matrix
    :param km: neuron half-activation points
    :param signal_sigma: level of noise in the signal
    :param nu: The inhibition coefficient
    :return: the result of the dynamics for the given input variables
    """
    noise = np.random.normal(scale=signal_sigma, size=y0.shape)
    input_signal = (J @ (y0 + noise))
    # input_signal += noise
    input_signal[input_signal < 0] = 0
    input_signal[input_signal > 1] = 1
    resp = ei_gain_func(input_signal, c.SEP_N, km, nu)
    return (-y0 + resp) / c.TAU


def separatrics_model_logistic(y0, t, J, km, signal_sigma, nu):
    """
    Function for running separatrix with logistic activation function in euler.
    :param y0: neuron activation
    :param t: time point
    :param J: connectivity matrix
    :param km: neuron half-activation points
    :param signal_sigma: level of noise in the signal
    :return: the result of the dynamics for the given input variables
    """
    noise = np.random.normal(scale=signal_sigma, size=y0.shape)
    input_signal = (J @ (y0 + noise))
    # input_signal += noise
    input_signal[input_signal < 0] = 0
    input_signal[input_signal > 1] = 1
    resp = logistic_func(input_signal, c.SI_SEP_N, km, nu)
    return (-y0 + resp) / c.TAU


def euler(func, y0, time, *args):
    """
    Generic euler integrator
    :param func: teh function to perform euler on
    :param y0: the start point of the euler integration.
    :param time: vector of the timepoints in which to simulate the dynamics. dt is determined by time[1]-time[0]
    :param args: Additional arguments that the function requires.
    :return: The full simulation of the dynamics
    """
    y0, time = list(map(np.asarray, [y0, time]))
    dt = time[1] - time[0]
    results = np.zeros(time.shape + y0.shape)
    results[0, :] = y0
    for i in range(1, time.size):
        results[i, :] = results[i - 1, :] + dt * func(results[i - 1, :], i, *args)
    return results


def separatrix_run(args):
    """
    Run a separatrix simulation with a set of parameters.
    :param args: a tuple with the following variables on the following order:
        * Signal level to start the simulation with. Must be in [0,1]
        * Level of noise in the signal while evolving the dynamics
        * Level of heterogeneity in the half-activation points of neurons
        * The connectivity matrix of the neurons
        * The number of neurons in a population
        * the number of populations to run simultaneously
    :return: The full simulation of the dynamics
    """
    base_sig_level, signal_sigma, km_sigma, nu, connectivity_mat, n_neurons, n_repeats = args
    time = np.arange(0, c.SEP_MAX_T, c.SEP_DT)
    np.random.seed(c.SEED)
    km = np.random.uniform(low=0.5 - km_sigma, high=0.5 + km_sigma, size=(n_neurons, n_repeats))
    signal = base_sig_level + np.random.normal(scale=signal_sigma, size=(n_neurons, n_repeats))
    signal[signal < 0] = 0
    signal[signal > 1] = 1
    activity = euler(separatrics_model, signal, time,
                     connectivity_mat, km, signal_sigma, nu)  # (time,n_neuron,n_repeats)
    return activity.astype(np.float32)


def separatrics_run_logistic(args):
    """
    Same as separatrix_run, only with the logistic activation function
    """
    base_sig_level, signal_sigma, km_sigma, nu, connectivity_mat, n_neurons, n_repeats = args
    time = np.arange(0, c.SEP_MAX_T, c.SEP_DT)
    np.random.seed(c.SEED)
    km = np.random.uniform(low=0.5 - km_sigma, high=0.5 + km_sigma, size=(n_neurons, n_repeats))
    signal = base_sig_level + np.random.normal(scale=signal_sigma, size=(n_neurons, n_repeats))
    signal[signal < 0] = 0
    signal[signal > 1] = 1
    activity = euler(separatrics_model_logistic, signal, time,
                     connectivity_mat, km, signal_sigma, nu)
    return activity


def logistic_func(x, slope=35, threshold=0.5, nu=1.):
    """
    Logistic function
    :param x: the value/s to perform the function on
    :param slope: the slope of the logistic function (beta)
    :param threshold: the half-activation point of the logistic function
    """
    return 1 / (nu + np.exp(-slope * (x - threshold)))


def ei_logistic_func(s, slope, threshold, nu):
    """
    E/I model with logistic function
    :param s: Signal level
    :param slope: the slope of the logistic function (beta)
    :param threshold: the half-activation point of the logistic function
    :param nu: The scaling of inhibition. nu<1 implies decreased inhibition
    """
    return np.exp(slope * (s - threshold)) / (nu + np.exp(slope * (s - threshold)))


def run_learning_trial(n=None, nu=None):
    """
    Run a hebbian learning trial
    :param n: The slopes of the neuronal populations
    :return: The Km over time
    """
    if n is None:
        n = [c.LR_NT_N, c.LR_NT_N, c.LR_ASD_N]
    if nu is None:
        nu = [1., c.EI_NU, 1.]

    n = np.array(n)
    nu = np.array(nu)
    Km = np.zeros((c.LR_MAX_T + 1, n.size)) + c.LR_START_THRESHOLD
    responses = np.zeros((c.LR_MAX_T + 1, n.size))
    max_resp = ei_gain_func(1, n, Km[0, :], nu)
    for i in range(1, c.LR_MAX_T + 1):
        s = np.random.uniform(0, 1)
        responses[i, :] = ei_gain_func(s, n, Km[i - 1, :], nu)
        category = 1 * (s > c.LR_THRESHOLD)
        is_correct = (responses[i, :] > c.LR_THRESHOLD) == category
        Km[i, :] = Km[i - 1, :] - (1 - is_correct) * c.LR_ALPHA * (category * max_resp - responses[i, :])
    return Km


def run_learning_trial_logistic(n=None):
    """
    Run a hebbian learning trial with logistic activation function
    :param n: The slopes of the neuronal populations
    :return: The Km over time
    """
    if n is None:
        n = [c.SI_LR_NT_N, c.SI_LR_ASD_N]
    n = np.array(n)
    Km = np.zeros((c.LR_MAX_T + 1, n.size))
    responses = np.zeros((c.LR_MAX_T + 1, n.size))
    for i in range(1, c.LR_MAX_T + 1):
        s = np.random.uniform(0, 1)
        responses[i, :] = logistic_func(s, n, Km[i - 1, :])
        exp_val = 1 * (s > 0.5)
        Km[i, :] = Km[i - 1, :] - c.LR_ALPHA * (exp_val - responses[i, :])
    return Km


def num2print(n):
    """
    Formats a number for printing. Below 1e-4 formats as scientific with precision of 1.
    :param n: The number to print
    :return: The formatted number
    """
    try:
        iter(n)
        ret_arr = np.zeros_like(n, float)
        for i in range(ret_arr.size):
            if n[i] == 0:
                ret_arr[i] = 0
            if n[i] <= 1e-4:
                ret_arr[i] = np.format_float_scientific(n[i], 1)
            else:
                current_num = abs(n[i]) * 10
                round_value = 1
                while not (current_num // 1):
                    current_num *= 10
                    round_value += 1
                ret_arr[i] = round(n[i], round_value)
        return ret_arr

    except:
        if n == 0:
            return 0
        if n <= 1e-4:
            return np.format_float_scientific(n, 1)
        current_num = abs(n) * 10
        round_value = 1

        while not (current_num // 1):
            current_num *= 10
            round_value += 1
        return round(n, round_value)


def num2latex(n):
    if n == 0:
        return "$1e^{-5}$"
    if n <= 1e-3:
        return "$" + re.sub(r"([-+]\d+)", r"^{\1}", str(np.format_float_scientific(n, 1))) + "$"
    return "$" + str(np.round(n, 4)) + "$"


def shift_function(x, y, return_dist=False):
    """
    Calculates the shift function for First vector - Second vector
    :param x: First vector
    :param y: Second vector
    :param return_dist: bool, if True returns the decile difference bootstrap distribution
    :return: mean decile difference, low ci, high ci
    """
    num_boot = 20000
    boot_x = np.random.choice(x, size=(num_boot, x.size))
    boot_y = np.random.choice(y, size=(num_boot, y.size))
    deciles = np.arange(0.1, 1, 0.1)
    x_deciles = np.quantile(boot_x, deciles, axis=1)
    y_deciles = np.quantile(boot_y, deciles, axis=1)
    decile_diff = x_deciles - y_deciles
    decile_diff_mean = decile_diff.mean(axis=1)
    diff_low_ci, diff_high_ci = np.quantile(decile_diff, [0.025, 0.975], axis=1)
    if return_dist:
        return decile_diff_mean, diff_low_ci, diff_high_ci, decile_diff
    return decile_diff_mean, diff_low_ci, diff_high_ci


def get_threshold_pass_idx(data, thresh):
    """
    calculates the index in which a vector passes a threshold
    :param data: The vector
    :param thresh: The threshold
    :return: array of indices
    """
    return np.argmax(data >= thresh), np.argwhere(data >= thresh)[-1][0]


def get_width_of_var(variances_list, s):
    """
    Calculate the width of the variances
    """
    variances_list = [var for var in variances_list if var is not None]
    n_vars = len(variances_list)
    start_sig = [[[] for j in range(n_vars)] for i in range(2)]
    end_sig = [[[] for j in range(n_vars)] for i in range(2)]
    for i, var in enumerate(variances_list):
        for j, thresh in enumerate([var.max() / 2, var.max() / np.e]):
            st_idx, e_idx = get_threshold_pass_idx(var, thresh)
            start_sig[j][i].append(s[st_idx])
            end_sig[j][i].append(s[e_idx])
    return np.squeeze(np.array(end_sig)) - np.squeeze(np.array(start_sig))


def get_effective_n(population_resp, signal, km=0.5, func=hill_func):
    """
    Fit the slope of the population response from the response, signal and half-activation points
    :param population_resp: The response
    :param signal: The signal eliciting the response
    :param km: The half-activation points
    :param func: The function used to generate the population response
    :return: the fitted slope
    """
    return curve_fit(lambda S, n: func(S, n, km), np.squeeze(signal), population_resp)[0].item()


HETEROGENEITY_TO_N = None
HETEROGENEITY_TO_N_STD = None


def get_effective_n_from_heterogeneity(heterogeneity, base_n=16):
    global HETEROGENEITY_TO_N, HETEROGENEITY_TO_N_STD
    if HETEROGENEITY_TO_N is None:
        N_NEURONS = 300
        N = base_n
        N_LEVELS = 500
        REPEATS = 100
        s = np.linspace(0, 1, N_LEVELS)[:, None]
        noise_level_list = np.linspace(0, 0.5, N_LEVELS)

        # retrieved_n = np.zeros_like(noise_level_list, dtype=np.float64)

        def simulate_an_retrieve_effective_n(s, noise_level):
            ns = np.zeros(REPEATS)
            for i in range(REPEATS):
                km = 0.5 + noise_level * np.random.uniform(-1, 1, size=(1, N_NEURONS))
                resp = hill_func(s, N, km)
                ns[i] = curve_fit(lambda S, n: hill_func(S, n, 0.5), np.squeeze(s), resp.mean(1))[0].item()
            return [ns.mean(), ns.std()]

        retrieved_n = []
        retrieved_n_std = []
        for noise_level in tqdm(noise_level_list, desc="Heterogeneity to N:"):
            n, std = simulate_an_retrieve_effective_n(s, noise_level)
            retrieved_n.append(n)
            retrieved_n_std.append(std)

        retrieved_n = np.array(retrieved_n)
        retrieved_n_std = np.array(retrieved_n_std)
        HETEROGENEITY_TO_N = np.vstack([noise_level_list, retrieved_n])
        HETEROGENEITY_TO_N_STD = np.vstack([noise_level_list, retrieved_n_std])
    insert_idx = np.minimum(np.searchsorted(HETEROGENEITY_TO_N[0], heterogeneity), HETEROGENEITY_TO_N.shape[-1] - 1)
    if isinstance(insert_idx, np.ndarray):
        insert_idx[insert_idx == 0] = 1
    elif insert_idx == 0:
        insert_idx = 1
    diff_left, diff_right = np.abs(heterogeneity - HETEROGENEITY_TO_N[0, insert_idx - 1]), np.abs(
        heterogeneity - HETEROGENEITY_TO_N[0, insert_idx])
    return (diff_left / (diff_left + diff_right)) * HETEROGENEITY_TO_N[1, insert_idx - 1] + \
        (diff_right / (diff_left + diff_right)) * HETEROGENEITY_TO_N[1, insert_idx]


def get_n_heterogeneity(n):
    if HETEROGENEITY_TO_N is None:
        _ = get_effective_n_from_heterogeneity(0.1)
    return HETEROGENEITY_TO_N_STD[
        1, np.minimum(np.searchsorted(-HETEROGENEITY_TO_N[1], -n), HETEROGENEITY_TO_N_STD.shape[-1] - 1)]


LOG_HETEROGENEITY_TO_N = None


def get_effective_log_n_from_heterogeneity(heterogeneity):
    global LOG_HETEROGENEITY_TO_N
    if LOG_HETEROGENEITY_TO_N is None:
        N_NEURONS = 300
        N = 40
        N_LEVELS = 500
        REPEATS = 100
        s = np.linspace(0, 1, N_LEVELS)[:, None]
        noise_level_list = np.linspace(0, 0.5, N_LEVELS)

        # retrieved_n = np.zeros_like(noise_level_list, dtype=np.float64)

        def simulate_an_retrieve_effective_n(s, noise_level):
            retrieved_n = 0
            for _ in range(REPEATS):
                km = 0.5 + noise_level * np.random.uniform(-1, 1, size=(1, N_NEURONS))
                resp = logistic_func(s, N, km)
                retrieved_n += curve_fit(lambda S, n: logistic_func(S, n, 0.5), np.squeeze(s), resp.mean(1))[0].item()
            retrieved_n /= REPEATS
            return retrieved_n

        retrieved_n = []
        for noise_level in tqdm(noise_level_list, desc="Heterogeneity to N:"):
            retrieved_n.append(simulate_an_retrieve_effective_n(s, noise_level))
        retrieved_n = np.array(retrieved_n)
        LOG_HETEROGENEITY_TO_N = np.vstack([noise_level_list, retrieved_n])
    insert_idx = np.searchsorted(LOG_HETEROGENEITY_TO_N[0], heterogeneity)
    diff_left, diff_right = np.abs(heterogeneity - LOG_HETEROGENEITY_TO_N[0, insert_idx - 1]), np.abs(
        heterogeneity - LOG_HETEROGENEITY_TO_N[0, insert_idx])
    return (diff_left / (diff_left + diff_right)) * LOG_HETEROGENEITY_TO_N[1, insert_idx - 1] + \
        (diff_right / (diff_left + diff_right)) * LOG_HETEROGENEITY_TO_N[1, insert_idx]


N_TO_HETEROGENEITY = dict()


def get_heterogeneity_from_n(n, base_n, n_neurons=300, n_levels=200, repeats=20):
    global N_TO_HETEROGENEITY
    current_key = f"{base_n}"
    if n == base_n:
        return 0
    if current_key not in N_TO_HETEROGENEITY.keys():
        s = np.linspace(0, 1, n_levels)[:, None]
        noise_level_list = np.linspace(0, 0.5, n_levels)

        # retrieved_n = np.zeros_like(noise_level_list, dtype=np.float64)

        def simulate_an_retrieve_effective_n(s, noise_level):
            retrieved_n = 0
            for _ in range(repeats):
                km = 0.5 + noise_level * np.random.uniform(-1, 1, size=(1, n_neurons))
                resp = hill_func(s, base_n, km)
                retrieved_n += curve_fit(lambda S, n: hill_func(S, n, 0.5), np.squeeze(s), resp.mean(1))[0].item()
            retrieved_n /= repeats
            return retrieved_n

        retrieved_n = []
        for noise_level in noise_level_list:
            retrieved_n.append(simulate_an_retrieve_effective_n(s, noise_level))
        retrieved_n = np.array(retrieved_n)
        N_TO_HETEROGENEITY[current_key] = np.vstack([noise_level_list, retrieved_n])

    cur_n_arr = N_TO_HETEROGENEITY[current_key]
    insert_idx = np.searchsorted(-cur_n_arr[1], -n)
    if insert_idx == cur_n_arr.shape[1]:
        return cur_n_arr[0, insert_idx - 1]
    diff_left, diff_right = np.abs(n - cur_n_arr[1, insert_idx - 1]), np.abs(
        n - cur_n_arr[1, insert_idx])
    return (diff_left / (diff_left + diff_right)) * cur_n_arr[0, insert_idx - 1] + \
        (diff_right / (diff_left + diff_right)) * cur_n_arr[0, insert_idx]


def permutation_test(x: np.array, y: np.array, test_func, alternative="two.sided", n_perm=10000,
                     return_dist=False, center_dists=False, plot=False, hist_color=None, line_color=None):
    """
    Perform permutation test of the given statistic with the data.
    :param x: 1D array representing group 1
    :param y: 1D array representing group 2
    :param test_func: Callable that gets two 1D arrays and calculates a statistic on them, Used to build H0.
    :param alternative: The hypothesis direction:
        * two.sided (Group 1 != Group 2)
        * greater (Group 1 > Group 2)
        * less (Group 1 < Group 2)
    :param n_perm: number of permutations to perform. Default 10000
    :param return_dist: boolean, whether to return the generated H0 distribution. Default False.
    :param plot: boolean, whether to plot the statistic with the generated H0 distribution. Default False.
    :param center_dists: boolean, whether to subtract the mean of each group. Default False.
    Should be done if equal means are assumed.
    :return: p value of the permutation test, (optional - h0 distribution), (optional - test figure)
    """
    if alternative not in {"two.sided", "greater", "less"}:
        raise AssertionError(f"alternative must be two.sided, greater, or less, yet it is {alternative}")
    if center_dists:
        pooled_data = np.concatenate([x - x.mean(), y - y.mean()]).flatten()
    else:
        pooled_data = np.concatenate([x, y]).flatten()
    permutation_h0 = np.zeros(n_perm, dtype=np.float64)
    for i in range(n_perm):
        permutation_data = pooled_data[np.random.permutation(pooled_data.size)]
        permutation_h0[i] = test_func(permutation_data[:x.size], permutation_data[x.size:])
    p = None
    statistic = test_func(x, y)
    if alternative == "two.sided":
        p = 1 - (np.abs(permutation_h0) <= np.abs(statistic)).mean()
    elif alternative == "greater":
        p = 1 - (permutation_h0 <= statistic).mean()
    elif alternative == "less":
        p = 1 - (statistic <= permutation_h0).mean()
    return_vals = (p,)
    if return_dist:
        return_vals = return_vals + (permutation_h0,)
    if plot:
        fig = pl.plot_hist_with_stat(statistic, permutation_h0, "Permutation test", p=p if p > 0 else 1 / n_perm,
                                     hist_color=hist_color,
                                     line_color=line_color)
        return_vals = return_vals + (fig,)
    return return_vals


def power_analysis(group1, group2, test_func, population_sizes=None, n_boot=10000, seed=c.SEED, **kwargs):
    """
    Perform power analysis of the given statistic with the data.
    :param group1: 1D array representing group 1
    :param group2: 1D array representing group 2
    :param test_func: Callable that gets two 2D arrays and calculates a statistic and it's p value on them, Used to build H0.
    :param population_sizes: list of population sizes to test
    :param n_boot: number of bootstrap samples to perform. Default 10000
    :return: power of the test for each population size
    """
    if seed is not None:
        np.random.seed(seed)
    if population_sizes is None:
        population_sizes = group1.size // 5
    if isinstance(population_sizes, int):
        population_sizes = np.unique(np.linspace(2, group1.size, population_sizes, dtype=np.int64))
        if group1.size not in population_sizes:
            population_sizes = np.append(population_sizes, group1.size)
    elif isinstance(population_sizes, np.ndarray):
        population_sizes = np.unique(population_sizes).astype(int)
        if group1.size not in population_sizes:
            population_sizes = np.append(population_sizes, group1.size)
    power = np.zeros_like(population_sizes, dtype=np.float64)
    for i, pop_size in enumerate(tqdm(population_sizes, desc="Power analysis progress")):
        group1_boot_idx = np.random.choice(np.arange(group1.size), size=(n_boot, pop_size), replace=True)
        group2_boot_idx = np.random.choice(np.arange(group2.size), size=(n_boot, pop_size), replace=True)
        power[i] = (test_func(group1[group1_boot_idx], group2[group2_boot_idx], axis=-1, **kwargs)[1] < 0.05).mean()
    return population_sizes, power


def cohens_d(x, y):
    nx = x.shape[-1]
    ny = y.shape[-1]
    dof = nx + ny - 2
    return (np.mean(x, axis=-1) - np.mean(y, axis=-1)) / np.sqrt(
        ((nx - 1) * np.std(x, axis=-1, ddof=1) ** 2 + (ny - 1) * np.std(y, axis=-1, ddof=1) ** 2) / dof)


def adjacency_effect_size_calc(means, stds):
    return (np.diff(means)) / (stds[1:] + stds[:-1])


def gain_func_generator(n, c=1., **kwargs):
    return lambda x, y=n, z=c: ei_gain_func(x, y, 0.5, z)


def rosenberg_gain_func_generator(c=1., **kwargs):
    return lambda x, y=c: ei_gain_func(x, 1, 0.5, y)


def sample_truncnorm(loc, scale, low, high, size):
    return tn(a=(low - loc) / scale, b=(high - loc) / scale, loc=loc, scale=scale).rvs(size=size)


def load_power_analysis_data(nt_n, asd_n):
    data = dict()
    individual_ns = dict()
    data_path = os.path.join('data', 'data_%d_%d' % (nt_n, asd_n))
    for f in os.listdir(data_path):
        if f.endswith('.npz') and "power_analysis" in f:
            sim_type = f[:f.find("_")]
            het_level = float(f[f.rfind("_") + 1:f.rfind(".")])
            loaded = np.load(os.path.join(data_path, f), allow_pickle=True)
            if data.get(het_level, None) is None:
                data[het_level] = dict()
            data[het_level][sim_type] = dict()
            for k in loaded.keys():
                data[het_level][sim_type][k] = loaded[k]
        elif f.endswith('.npz'):
            sim_type = f[:f.find("_")]
            het_level = float(f[f.rfind("_") + 1:f.rfind(".")])
            loaded = np.load(os.path.join(data_path, f))
            if individual_ns.get(het_level, None) is None:
                individual_ns[het_level] = dict()
            individual_ns[het_level][sim_type] = dict()
            for k in loaded.keys():
                individual_ns[het_level][sim_type][k] = loaded[k]
    return data, individual_ns


def parse_power_analysis_data(data, individual_ns, return_n=False):
    population = list(data.values())[0]["br"]["pop"]
    shape = (5, len(data.keys()), population.size)
    powers = np.zeros(shape, dtype=float)
    ns_shape = (4, len(data.keys()), 3000)
    asd_ns = np.zeros(ns_shape, dtype=float)
    nt_ns = np.zeros(ns_shape, dtype=float)
    heterogeneities = np.array(sorted(data.keys()))
    for i, k in enumerate(heterogeneities):
        powers[0, i, :] = data[k]["br"]["power_ratio"]
        powers[1, i, :] = data[k]["encoding"]["power"]
        powers[2, i, :] = data[k]["slow"]["power"]
        try:
            powers[3, i, :] = data[k]["lca"]["power"][0]
        except Exception:
            powers[3, i, :] = data[k]["lca"]["power"][0][:-1]
        powers[4, i, :] = data[k]["lca"]["power"][1]

        asd_ns[0, i, :] = np.squeeze(individual_ns[k]["br"]["asd"])
        asd_ns[1, i, :] = np.squeeze(individual_ns[k]["encoding"]["asd"])
        asd_ns[2, i, :] = np.squeeze(individual_ns[k]["slow"]["asd"])
        asd_ns[3, i, :] = np.squeeze(individual_ns[k]["lca"]["asd"])
        nt_ns[0, i, :] = np.squeeze(individual_ns[k]["br"]["nt"])
        nt_ns[1, i, :] = np.squeeze(individual_ns[k]["encoding"]["nt"])
        nt_ns[2, i, :] = np.squeeze(individual_ns[k]["slow"]["nt"])
        nt_ns[3, i, :] = np.squeeze(individual_ns[k]["lca"]["nt"])
    if return_n:
        return population, powers, heterogeneities, asd_ns, nt_ns
    return population, powers, heterogeneities


def get_tapping_heterogeneities():
    if not os.path.exists("individual_fits_ns.npz"):
        fitting_kwargs = dict(prior_var=0.1, perturb_prob=1e-6, base_n=20, n_neurons=200, perceptual_noise=0.03,
                              fit_scale_factor=50)
        asd_ns = TappingData("ASD.mat", **fitting_kwargs).get_hill_coefficients()
        nt_ns = TappingData("NT.mat", **fitting_kwargs).get_hill_coefficients()
        np.savez("individual_fits_ns.npz", asd_ns=asd_ns, nt_ns=nt_ns)
    else:
        loaded = np.load("individual_fits_ns.npz")
        asd_ns = loaded["asd_ns"]
        nt_ns = loaded["nt_ns"]
    asd_hets = np.array([get_heterogeneity_from_n(n, 20) for n in asd_ns])
    nt_hets = np.array([get_heterogeneity_from_n(n, 20) for n in nt_ns])
    return asd_hets, asd_ns, nt_hets, nt_ns


def tapas_sgm(x, a):
    return a / (1 + np.exp(-x))


def load_hgf(path=os.path.join('data', 'hgf.mat')):
    hgf_mat = scio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return hgf_mat["asd_fit"], hgf_mat["sim_fit"], hgf_mat["alphas"], hgf_mat["omegas"]


def mse_bic(mse, k, n):
    return k * np.log(n) - 2 * np.log(mse)


def cm2inch(cm):
    return cm / 2.54


def cm_to_user_units(cm):
    return cm2inch(cm) * 72


def resize_plot(plot_name, target_width):
    """
    Resize a pdf plot to a target width
    :param plot_name: Path to the plot
    :param target_width: target width in cm
    """
    reader = PdfReader(plot_name)
    # copy pdf page 0 to new pdf, changing the size of the page to have the same ratio, but with width of 8.8cm

    writer = PdfWriter()
    new_width = cm_to_user_units(target_width)
    for page in reader.pages:
        scale = new_width / float(page.mediabox[-2])
        page.scale_by(scale)
        page.add_transformation(Transformation().scale(sx=1, sy=1))
        writer.add_page(page)

    # save the writer to disk
    with open(plot_name[:-4]+f"_resize_to_{target_width}.pdf", "wb") as f:
        writer.write(f)
