from scipy.io import loadmat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from scipy.optimize import curve_fit
from scipy.interpolate import Akima1DInterpolator
import scipy
import itertools
import os

HETEROGENEITY_TO_N = dict()


# %% fitting

def coarse_to_fine_search(x: np.array, y: [int, float, np.array], fit_params_dict: dict, fitting_func: callable,
                          loss_func: callable, steps: int = 7,
                          iterations: int = 3, return_loss=False, rtol=1.e-5, atol=1.e-8, **kwargs):
    params = fit_params_dict.copy()
    losses = np.zeros((iterations, *[steps for _ in range(len(params.keys()))]))
    iter_step_size = np.prod(losses.shape[1:])
    last_loss = np.inf
    if kwargs.get("tqdm", True):
        iter_range = tqdm(range(iterations), desc="Iterations", leave=True)
    else:
        iter_range = range(iterations)
    for i in iter_range:
        iter_params = {k: np.linspace(v[0], v[1], steps) for k, v in params.items()}
        keys = list(iter_params.keys())
        values = list(itertools.product(*iter_params.values()))
        for j in range(len(values)):
            fit_result = fitting_func(x, **{keys[k]: values[j][k] for k in range(len(keys))}, **kwargs)
            idx = np.unravel_index(i * iter_step_size + j, losses.shape)
            losses[idx] = loss_func(y, fit_result, **kwargs)
        min_loss_idx = np.unravel_index((i * iter_step_size) + np.nanargmin(losses[i]), losses.shape)[1:]

        for k in range(len(keys)):
            cur_min_idx = min_loss_idx[k]
            cur_values = iter_params[keys[k]]
            min_idx, max_idx = cur_min_idx - 1, cur_min_idx + 1
            if cur_min_idx == steps - 1:
                max_idx = steps - 1
            elif cur_min_idx == 0:
                min_idx = 0
            params[keys[k]] = [cur_values[min_idx], cur_values[max_idx]]
        last_loss = losses[(i,) + min_loss_idx]
    best_params = {k: v[-1] if v[-1] == params[k][-1] else (v[0] if v[0] == params[k][0] else np.mean(v)) for k, v in
                   params.items()}
    if return_loss:
        return best_params, last_loss, losses
    return best_params, last_loss


# %% encoding capacity
def simple_loss(a, b, **kwargs):
    return np.mean((a - b) ** 2)


def total_encoding_capacity(n, k, v):
    kn = k ** n
    one_over_kn = 1 / kn
    return (0.5 * one_over_kn) * ((kn * (kn + (2 * kn * n) + v + (n * v))) / ((kn + v) ** 2) + (
            ((1 + n) * scipy.special.hyp2f1(1, (n - 1) / n, 2 - (1 / n), -v * one_over_kn)) / (n - 1)))


def get_normal_dist_from_quants(alpha, low_value, high_value):
    quantiles = [alpha / 2, 1 - alpha / 2]
    ppf = scipy.stats.norm().ppf(quantiles)
    scale = np.diff([low_value, high_value]) / np.diff(ppf)
    loc = (low_value + high_value) / 2
    return {"loc": loc, "scale": scale}


# %% tapping
def hill_func(s, n, k, nu=1.):
    s = np.clip(s, 0, 1)
    return (s ** n) / (nu * (s ** n) + k ** n)


def get_heterogeneity_from_n(n, base_n, nu):
    global HETEROGENEITY_TO_N
    current_key = f"{base_n}"
    if n == base_n:
        return 0
    if current_key not in HETEROGENEITY_TO_N.keys():
        N_NEURONS = 300
        N_LEVELS = 200
        REPEATS = 20
        s = np.linspace(0, 1, N_LEVELS)[:, None]
        noise_level_list = np.linspace(0, 0.5, N_LEVELS)

        # retrieved_n = np.zeros_like(noise_level_list, dtype=np.float64)

        def simulate_an_retrieve_effective_n(s, noise_level):
            retrieved_n = 0
            for _ in range(REPEATS):
                km = 0.5 + noise_level * np.random.uniform(-1, 1, size=(1, N_NEURONS))
                resp = hill_func(s, base_n, km)
                retrieved_n += curve_fit(lambda S, n: hill_func(S, n, 0.5), np.squeeze(s), resp.mean(1))[0].item()
            retrieved_n /= REPEATS
            return retrieved_n

        retrieved_n = []
        for noise_level in noise_level_list:
            retrieved_n.append(simulate_an_retrieve_effective_n(s, noise_level))
        retrieved_n = np.array(retrieved_n)
        HETEROGENEITY_TO_N[current_key] = np.vstack([noise_level_list, retrieved_n])

    cur_n_arr = HETEROGENEITY_TO_N[current_key]
    insert_idx = np.searchsorted(-cur_n_arr[1], -n)
    if insert_idx == cur_n_arr.shape[1]:
        return cur_n_arr[0, insert_idx - 1]
    diff_left, diff_right = np.abs(n - cur_n_arr[1, insert_idx - 1]), np.abs(
        n - cur_n_arr[1, insert_idx])
    return (diff_left / (diff_left + diff_right)) * cur_n_arr[0, insert_idx - 1] + \
        (diff_right / (diff_left + diff_right)) * cur_n_arr[0, insert_idx]


class TappingData:
    MAX_CHANGE = 45

    def __init__(self, filename, pre_lag=2, post_lag=7, half_range=0.3, e=None, r=None, s=None, d=None,
                 **fitting_kwargs) -> None:
        super().__init__()
        self.simulated_dynamics_loss = None
        self.fitted_nu = None
        self.half_range = half_range
        if filename:
            self._mat = loadmat(Path("data") / "tapping" / filename)
            self.e = self._reshape_loaded_array(self._mat['e'])
            self.r = self._reshape_loaded_array(self._mat['r'])
            self.s = np.round(self._reshape_loaded_array(self._mat['s']))
            self.d = self.r + np.pad(self.e, ((0, 0), (0, 0), (0, 0), (1, 0)))[..., :-1]
        else:
            self._mat = None
            self.e = e.copy()
            self.r = r.copy()
            self.s = s.copy()
            self.d = d.copy()
        self.pre_lag = pre_lag
        self.post_lag = post_lag
        self.n_subjects = self.e.shape[0]
        self.__subjects = [
            SubjectTappingData(self, i, filename[:-4] if filename else "no_name", self.pre_lag, self.post_lag,
                               **fitting_kwargs) for
            i in range(self.n_subjects)]
        self.fitting_kwargs = fitting_kwargs
        self.fitted_n = None
        self.rosenberg_fitted_nu = None
        self.group_dynamics_mean = None
        self.group_dynamics_std = None
        self.diffs = None
        self.lags = None
        self.simulated_dynamics_mean = None
        self.__current_index = 0

    def __len__(self):
        return len(self.subjects)

    def __iter__(self):
        return self

    def __next__(self):
        if self.__current_index < len(self.subjects):
            ret = self.subjects[self.__current_index]
            self.__current_index += 1
            return ret
        self.__current_index = 0
        raise StopIteration

    def __getitem__(self, i):
        return self.subjects[i]

    @property
    def subjects(self):
        return self.__subjects

    @property
    def n(self):
        return np.nanmean(self.fitted_n)

    def _reshape_loaded_array(self, arr):
        max_shape = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    max_shape = max(max_shape, arr[i, j, k].size)

        out = np.full(arr.shape + (max_shape,), np.nan)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    out[i, j, k, :arr[i, j, k].size] = np.squeeze(arr[i, j, k])
        return out

    def get_change_dynamics(self, pre_lag=2, post_lag=7):
        subj1_dyn, diffs, lags = self.subjects[0].get_change_dynamics(pre_lag, post_lag, None)
        change_dyn = [subj.get_change_dynamics(pre_lag, post_lag, None)[0] for subj in
                      tqdm(self.subjects, desc="Subjects")]
        return np.array(change_dyn), diffs, lags

    def plot_change_dynamics(self, pre_lag=2, post_lag=7):
        change_dyn, diffs, lags = self.get_change_dynamics(pre_lag, post_lag)
        plot_change_dynamics_helper(self.s, change_dyn, diffs, lags)

    def _get_hill_helper(self, idx):
        return self.subjects[idx].n

    def get_hill_coefficients(self):
        _ = [subj.n for subj in tqdm(self.subjects, desc="Subject")]
        coeffs = np.concatenate([subj._fitted_n.ravel() for subj in self.subjects])
        return coeffs[~np.isnan(coeffs)]

    def get_median_hill_coefficients(self):
        _ = [subj.n for subj in tqdm(self.subjects, desc="Subject")]
        meds = np.array([np.nanmedian(subj._fitted_n) for subj in self.subjects])
        return meds[~np.isnan(meds)]

    def fit_individual_subjects_ei(self, min_n=8., max_n=None, nu_range=[.1, .8]):
        if max_n is None:
            max_n = self.fitting_kwargs['base_n'] * 0.95
        _ = [subj.fit_ei_n_per_block(min_n, max_n, nu_range) for subj in tqdm(self.subjects, desc="Subject")]
        meds = np.array([np.nanmedian(subj._fitted_nu) for subj in self.subjects])
        return meds[~np.isnan(meds)]

    def get_median_inhibition_factors(self):
        _ = [subj.nu for subj in tqdm(self.subjects, desc="Subject")]
        meds = np.array([np.nanmedian(subj._fitted_nu) for subj in self.subjects])
        return meds[~np.isnan(meds)]

    def fit_to_group_dynamics(self, min_n=2., max_n=None, iterations=10):
        self._calculate_clean_dynamics()
        increased_freq_factor, new_x, true_freq = self._init_group_fit_output_vars()
        np.random.seed(97)
        for block in range(2, self.group_dynamics_mean.shape[0]):
            for direction in range(2):
                interpolated_dynamics, signal = self._prepare_block_fit(block, direction, increased_freq_factor, new_x,
                                                                        true_freq)
                if max_n is None:
                    max_n = self.fitting_kwargs['base_n'] * 0.95
                params, loss = coarse_to_fine_search(signal, interpolated_dynamics,
                                                     {"n": [min_n, max_n]},
                                                     self._kalman_filter, self._simple_loss, iterations=iterations,
                                                     block=block, **self.fitting_kwargs, rtol=0, atol=0,
                                                     post_lag=7, pre_lag=2, tqdm=False, seed=97)
                self.simulated_dynamics_loss[block, direction] = loss
                self.fitted_n[block, direction] = params["n"]
                self.simulated_dynamics_mean[block, direction] = \
                    self._kalman_filter(signal, params["n"], seed=97, **self.fitting_kwargs)[
                        [(i * increased_freq_factor) for i in range(self.group_dynamics_mean.shape[-1])]]
        return self.simulated_dynamics_mean

    def fit_to_group_dynamics_rosenberg(self, iterations=10):
        self._calculate_clean_dynamics()
        increased_freq_factor, new_x, true_freq = self._init_group_fit_output_vars()
        self.rosenberg_fitted_nu = np.full(self.group_dynamics_mean.shape[:-1], np.nan)

        np.random.seed(97)
        for block in range(2, self.group_dynamics_mean.shape[0]):
            for direction in range(2):
                interpolated_dynamics, signal = self._prepare_block_fit(block, direction, increased_freq_factor, new_x,
                                                                        true_freq)
                params, loss = coarse_to_fine_search(signal, interpolated_dynamics,
                                                     {"nu": [0.1, 1.]},
                                                     self._kalman_filter_rosenberg, self._simple_loss,
                                                     iterations=iterations,
                                                     block=block, **self.fitting_kwargs, rtol=0, atol=0,
                                                     post_lag=7, pre_lag=2, tqdm=False, seed=97)
                self.simulated_dynamics_loss[block, direction] = loss
                self.rosenberg_fitted_nu[block, direction] = params["nu"]
                self.simulated_dynamics_mean[block, direction] = \
                    self._kalman_filter_rosenberg(signal, params["nu"], seed=97, **self.fitting_kwargs)[
                        [(i * increased_freq_factor) for i in range(self.group_dynamics_mean.shape[-1])]]
        return self.simulated_dynamics_mean

    def _init_group_fit_output_vars(self):
        self.fitted_n = np.full(self.group_dynamics_mean.shape[:-1], np.nan)
        self.simulated_dynamics_mean = np.zeros_like(self.group_dynamics_mean)
        self.simulated_dynamics_loss = np.full(self.group_dynamics_mean.shape[:-1], np.nan)
        increased_freq_factor = self.fitting_kwargs.get('fit_scale_factor', 50)
        new_x = np.linspace(-self.pre_lag, self.post_lag,
                            ((self.group_dynamics_mean.shape[-1] - 1) * increased_freq_factor) + 1)
        true_freq = np.full_like(new_x, np.nan)
        return increased_freq_factor, new_x, true_freq

    def fit_ei_to_group_dynamics(self, min_n=2., max_n=None, nu_range=[.1, 1.], iterations=10):
        self._calculate_clean_dynamics()
        increased_freq_factor, new_x, true_freq = self._init_group_fit_output_vars()
        if max_n is None:
            max_n = self.fitting_kwargs['base_n'] * 0.95
        np.random.seed(97)
        self.fitted_nu = np.full(self.group_dynamics_mean.shape[:-1], np.nan)
        for block in range(2, self.group_dynamics_mean.shape[0]):
            for direction in range(2):
                interpolated_dynamics, signal = self._prepare_block_fit(block, direction, increased_freq_factor, new_x,
                                                                        true_freq)
                params, loss = coarse_to_fine_search(signal, interpolated_dynamics,
                                                     {"n": [min_n, max_n], "nu": nu_range},
                                                     self._kalman_filter, self._simple_loss, iterations=iterations,
                                                     block=block, **self.fitting_kwargs, rtol=0, atol=0,
                                                     post_lag=7, pre_lag=2, tqdm=False, seed=97)
                self.simulated_dynamics_loss[block, direction] = loss
                self.fitted_n[block, direction] = params["n"]
                self.fitted_nu[block, direction] = params["nu"]
                self.simulated_dynamics_mean[block, direction] = \
                    self._kalman_filter(signal, params["n"], nu=params["nu"], seed=97, **self.fitting_kwargs)[
                        [(i * increased_freq_factor) for i in range(self.group_dynamics_mean.shape[-1])]]
        return self.simulated_dynamics_mean

    def _calculate_clean_dynamics(self):
        subj1_dyn, self.diffs, self.lags = self.subjects[0].dynamics, self.subjects[0].diffs, self.subjects[0].lags
        group_dynamics = np.stack([subj.dynamics for subj in self.subjects])
        # remove all the non-tracking dynamics
        acc = group_dynamics[:, :, 0:1, :] < 500
        dec = group_dynamics[:, :, 1:, :] > 500
        acc_ok = acc.any(axis=-1)
        dec_ok = dec.any(axis=-1)
        ok_idx = np.concatenate([acc_ok.copy(), dec_ok.copy()], axis=2)
        ok_idx = np.repeat(ok_idx[..., None], group_dynamics.shape[-1], axis=-1)
        pre_change_metronome = np.stack(
            [np.nanmax(self.s[:, 1:, 0, :], axis=-1), np.nanmin(self.s[:, 1:, 0, :], axis=-1)],
            axis=-1)
        group_dynamics[~ok_idx] = np.nan
        group_dynamics = group_dynamics - np.nanmean(group_dynamics[..., np.array(self.lags) < 0], axis=-1,
                                                     keepdims=True) + pre_change_metronome[:, :, :, None]
        self.group_dynamics_mean = np.nanmean(group_dynamics, axis=0)
        self.group_dynamics_std = np.nanstd(group_dynamics, axis=0)

    def _prepare_block_fit(self, block, direction, increased_freq_factor, new_x, true_freq):
        pre_change = self.group_dynamics_mean[block, direction, :2].mean()
        post_change = 500 + (((-1) ** (direction + 1)) * (self.diffs[block] / 2))
        true_freq[:(increased_freq_factor * self.pre_lag + 1)] = pre_change
        true_freq[(increased_freq_factor * self.pre_lag + 1):] = post_change
        spl = Akima1DInterpolator(self.lags, self.group_dynamics_mean[block, direction])
        interpolated_dynamics = spl(new_x)
        signal = np.repeat(true_freq, self.fitting_kwargs["n_neurons"]).reshape((new_x.size, -1))
        return interpolated_dynamics, signal

    def bootstrap_fit_to_group_dynamics(self, min_n=2., n_bootstraps=1000):
        subj1_dyn, self.diffs, self.lags = self.subjects[0].dynamics, self.subjects[0].diffs, self.subjects[0].lags
        group_dynamics = np.stack([subj.dynamics for subj in self.subjects])
        # remove all the non-tracking dynamics
        acc = group_dynamics[:, :, 0:1, :] < 500
        dec = group_dynamics[:, :, 1:, :] > 500
        acc_ok = acc.any(axis=-1)
        dec_ok = dec.any(axis=-1)
        ok_idx = np.concatenate([acc_ok.copy(), dec_ok.copy()], axis=2)
        ok_idx = np.repeat(ok_idx[..., None], group_dynamics.shape[-1], axis=-1)

        pre_change_metronome = np.stack(
            [np.nanmax(self.s[:, 1:, 0, :], axis=-1), np.nanmin(self.s[:, 1:, 0, :], axis=-1)],
            axis=-1)
        group_dynamics[~ok_idx] = np.nan
        group_dynamics = group_dynamics - np.nanmean(group_dynamics[..., np.array(self.lags) < 0], axis=-1,
                                                     keepdims=True) + pre_change_metronome[:, :, :, None]
        self.boot_fitted_n = np.full((n_bootstraps,) + self.group_dynamics_mean.shape[:-1], np.nan)
        increased_freq_factor = 50
        new_x = np.linspace(-self.pre_lag, self.post_lag,
                            ((self.group_dynamics_mean.shape[-1] - 1) * increased_freq_factor) + 1)
        true_freq = np.full_like(new_x, np.nan)
        np.random.seed(97)
        bootstrap_idx = np.random.choice(range(self.n_subjects), (n_bootstraps, self.n_subjects))
        for i in tqdm(range(n_bootstraps), desc="Bootstrapping subjects in fit"):
            group_dynamics_mean = np.nanmean(group_dynamics[bootstrap_idx[i]], axis=0)
            for block in range(2, self.group_dynamics_mean.shape[0]):
                for direction in range(2):
                    pre_change = 500 + (((-1) ** direction) * (self.diffs[block] / 2))
                    post_change = 500 + (((-1) ** (direction + 1)) * (self.diffs[block] / 2))
                    true_freq[:(increased_freq_factor * self.pre_lag + 1)] = pre_change
                    true_freq[(increased_freq_factor * self.pre_lag + 1):] = post_change
                    spl = Akima1DInterpolator(self.lags, group_dynamics_mean[block, direction])
                    interpolated_dynamics = spl(new_x)

                    signal = np.repeat(true_freq, self.fitting_kwargs["n_neurons"]).reshape((new_x.size, -1))
                    params, loss = coarse_to_fine_search(signal, interpolated_dynamics,
                                                         {"n": [min_n, self.fitting_kwargs['base_n'] * 0.95]},
                                                         self._kalman_filter, self._simple_loss, iterations=5,
                                                         block=block, **self.fitting_kwargs, rtol=0, atol=0,
                                                         post_lag=7, pre_lag=2, tqdm=False, seed=97)
                    self.boot_fitted_n[i, block, direction] = params["n"]
        return self.boot_fitted_n

    def _freq_to_input_signal(self, freq):
        scaling = (self.MAX_CHANGE / self.half_range)
        out = ((freq - 500) / scaling) + 0.5
        out[out < 0] = 0
        out[out > 1] = 1
        return out

    @lru_cache
    def _get_dense_response(self, n):
        dense_sig = np.linspace(0, 1, 10000)
        return hill_func(dense_sig, n, 0.5), dense_sig

    def _output_signal_to_freq(self, estimated_signal):
        scaling = (self.MAX_CHANGE / self.half_range)
        return np.round(((estimated_signal - 0.5) * scaling) + 500, 2)

    def _kalman_filter(self, signal, n, **kwargs):
        prior_var = self.fitting_kwargs["prior_var"]
        perturb_prob = self.fitting_kwargs["perturb_prob"]
        base_n = self.fitting_kwargs["base_n"]
        nu = kwargs.get("nu", 1.)
        seed = 97
        np.random.seed(seed)
        signal = self._freq_to_input_signal(signal)
        signal += np.random.normal(0, self.fitting_kwargs['perceptual_noise'], signal.shape)
        signal = np.clip(signal, 0, 1)
        noise_level = get_heterogeneity_from_n(n, base_n, nu)

        km = 0.5 + noise_level * np.random.uniform(-1, 1, size=(1, signal.shape[-1]))
        resp = hill_func(signal, base_n, km, nu)
        var = resp.var(-1)
        mean_resp = resp.mean(-1)

        estimated_var = np.zeros_like(var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
        estimated_var[0] = prior_var
        resp_estimate = np.zeros_like(var, dtype=np.float64)
        resp_estimate[0] = mean_resp[0]
        epsilon = perturb_prob * prior_var
        for i in range(1, resp.shape[0]):
            if np.isnan(mean_resp[i]):
                estimated_var[i] = estimated_var[i - 1]
                resp_estimate[i] = resp_estimate[i - 1]
                continue
            last_est_var = (1 - perturb_prob) * estimated_var[i - 1]
            kg = (last_est_var + epsilon) / ((last_est_var + epsilon) + var[i - 1])
            estimated_var[i] = (1 - kg) * (last_est_var + epsilon)
            resp_estimate[i] = resp_estimate[i - 1] + kg * (mean_resp[i] - resp_estimate[i - 1])

        dense_sig = np.linspace(0, 1, 10000)
        dense_resp = np.squeeze(hill_func(dense_sig[:, None], base_n, km, nu).mean(-1))
        sig_estimate = dense_sig[np.searchsorted(dense_resp, resp_estimate) - 1]

        return self._output_signal_to_freq(sig_estimate)

    def _kalman_filter_rosenberg(self, signal, nu, **kwargs):
        prior_var = self.fitting_kwargs["prior_var"]
        perturb_prob = self.fitting_kwargs["perturb_prob"]
        seed = 97
        np.random.seed(seed)
        signal = self._freq_to_input_signal(signal)
        signal += np.random.normal(0, self.fitting_kwargs['perceptual_noise'], signal.shape)
        signal = np.clip(signal, 0, 1)
        km = 0.5
        inhib = nu + np.random.uniform(-0.01, 0.01, size=(1, signal.shape[-1]))

        resp = hill_func(10 * signal, 1., km, inhib)
        var = resp.var(-1)
        mean_resp = resp.mean(-1)

        estimated_var = np.zeros_like(var, dtype=np.float64)  # shape=(SR_NUM_REPS, SR_NUM_STEPS)
        estimated_var[0] = prior_var
        resp_estimate = np.zeros_like(var, dtype=np.float64)
        resp_estimate[0] = mean_resp[0]
        epsilon = perturb_prob * prior_var
        for i in range(1, resp.shape[0]):
            if np.isnan(mean_resp[i]):
                estimated_var[i] = estimated_var[i - 1]
                resp_estimate[i] = resp_estimate[i - 1]
                continue
            last_est_var = (1 - perturb_prob) * estimated_var[i - 1]
            kg = (last_est_var + epsilon) / ((last_est_var + epsilon) + var[i - 1])
            estimated_var[i] = (1 - kg) * (last_est_var + epsilon)
            resp_estimate[i] = resp_estimate[i - 1] + kg * (mean_resp[i] - resp_estimate[i - 1])

        dense_sig = np.linspace(0, 1, 10000)
        dense_resp = np.squeeze(hill_func(10 * dense_sig[:, None], 1., km, inhib).mean(-1))
        sig_estimate = dense_sig[np.searchsorted(dense_resp, resp_estimate) - 1]

        return self._output_signal_to_freq(sig_estimate)

    def _simple_loss(self, real_resp, est_resp, **kwargs):
        increased_freq_factor = self.fitting_kwargs.get('fit_scale_factor', 50)
        return np.nanmean((real_resp[3 * increased_freq_factor:] - est_resp[3 * increased_freq_factor:]) ** 2)

    def _change_dynamics_loss(self, real_resp, est_resp, pre_lag, **kwargs):
        eval_idx = [i * 50 for i in range(10)]
        return np.nanmean((real_resp[eval_idx] - est_resp[eval_idx]) ** 2)


class SubjectTappingData:
    MAX_CHANGE = 45  # Hz

    def __init__(self, full_data: TappingData, subject_idx: int, name, pre_lag=2, post_lag=7,
                 **fitting_kwargs) -> None:
        super().__init__()
        self.e = full_data.e[subject_idx]
        self.r = full_data.r[subject_idx]
        self.s = full_data.s[subject_idx]
        self.d = full_data.d[subject_idx]
        self._kalman_filter = full_data._kalman_filter
        self._loss_func = full_data._change_dynamics_loss
        self._fitted_n = None
        self.nblocks, self.nreps, self.ntimes = self.s.shape
        self.pre_lag = pre_lag
        self.post_lag = post_lag
        self.dynamics, self.diffs, self.lags = self.get_change_dynamics()
        self.sim_dynamics = np.full_like(self.dynamics, np.nan)
        self.fitting_kwargs = fitting_kwargs
        self.name = name
        self.subject_idx = subject_idx

    @property
    def n(self):
        if self._fitted_n is None:
            self.fit_n()
        return np.nanmean(self._fitted_n)

    @property
    def nu(self):
        if self._fitted_nu is None:
            self.fit_ei_n_per_block()
        return np.nanmean(self._fitted_nu)

    def fit_n_per_block(self, min_n=2., seed=97) -> np.array:
        os.makedirs(f"data/{self.name}_fits", exist_ok=True)
        try:
            loaded = np.load(f"data/{self.name}_fits/{self.name}{str(self.subject_idx).zfill(3)}_" + "_".join(
                [f"{k}:{v:.2g}" for k, v in self.fitting_kwargs.items()]) + ".npz")
            if self.fitting_kwargs != loaded["fitting_kwargs"]:
                raise ValueError("Fitting kwargs don't match, refitting")
            self.sim_dynamics = loaded["sim_dynamics"]
            self._fitted_n = loaded["fitted_n"]
            self.simulated_dynamics_loss = loaded["simulated_dynamics_loss"]
            return self._fitted_n
        except:
            pass
        np.random.seed(seed)
        self._fitted_n = np.full(self.dynamics.shape[:-1], np.nan)
        increased_freq_factor = self.fitting_kwargs.get('fit_scale_factor', 50)
        new_x = np.linspace(-self.pre_lag, self.post_lag, ((self.dynamics.shape[-1] - 1) * increased_freq_factor) + 1)
        true_freq = np.full_like(new_x, np.nan)
        np.random.seed(97)
        self.simulated_dynamics_loss = np.full(self.dynamics.shape[:-1], np.nan)
        for block in range(2, self.dynamics.shape[0]):
            for direction in range(2):
                if np.isnan(self.dynamics[block, direction]).all():
                    continue

                pre_change = 500 + (((-1) ** direction) * (self.diffs[block] / 2))
                post_change = 500 + (((-1) ** (direction + 1)) * (self.diffs[block] / 2))
                comparator = np.greater if direction == 0 else np.less
                # if there was no update
                if comparator(self.dynamics[block, direction, self.pre_lag + 1:], 500).all():
                    continue
                # if the update happened before the change
                if (~comparator(self.dynamics[block, direction, :self.pre_lag + 1], 500)).any():
                    continue
                true_freq[:(increased_freq_factor * self.pre_lag + 1)] = pre_change
                true_freq[(increased_freq_factor * self.pre_lag + 1):] = post_change
                spl = Akima1DInterpolator(self.lags, self.dynamics[block, direction])
                interpolated_dynamics = spl(new_x)
                signal = np.repeat(true_freq, self.fitting_kwargs["n_neurons"]).reshape(
                    (new_x.size, -1))
                # signal += np.random.normal(0, np.sqrt(np.nanmean(self.group_dynamics_std[block, direction])), signal.shape)
                try:
                    params, loss = coarse_to_fine_search(signal, interpolated_dynamics,
                                                         {"n": [min_n, self.fitting_kwargs['base_n'] * 0.95]},
                                                         self._kalman_filter, self._simple_loss, iterations=10,
                                                         block=block, **self.fitting_kwargs, rtol=0, atol=0,
                                                         post_lag=7, pre_lag=2, tqdm=False, seed=97)
                except ValueError:
                    continue
                self.simulated_dynamics_loss[block, direction] = loss
                self._fitted_n[block, direction] = params["n"]
                self.sim_dynamics[block, direction] = \
                    self._kalman_filter(signal, params["n"], seed=seed, **self.fitting_kwargs)[
                        [(i * increased_freq_factor) for i in range(self.dynamics.shape[-1])]]
        np.savez(f"data/{self.name}_fits/{self.name}{str(self.subject_idx).zfill(3)}_" + "_".join(
            [f"{k}-{v:.2g}" for k, v in self.fitting_kwargs.items()]) + ".npz",
                 sim_dynamics=self.sim_dynamics, fitted_n=self._fitted_n,
                 simulated_dynamics_loss=self.simulated_dynamics_loss, fitting_kwargs=self.fitting_kwargs)
        return self._fitted_n

    def _simple_loss(self, real_resp, est_resp, **kwargs):
        increased_freq_factor = self.fitting_kwargs.get('fit_scale_factor', 50)
        return np.nanmean((real_resp[3 * increased_freq_factor:] - est_resp[3 * increased_freq_factor:]) ** 2)

    def fit_ei_n_per_block(self, min_n=12., max_n=None, nu_range=[.1, 1.], seed=97) -> np.array:
        os.makedirs(f"data/{self.name}_ei_fits", exist_ok=True)
        try:
            loaded = np.load(f"data/{self.name}_ei_fits/{self.name}{str(self.subject_idx).zfill(3)}_" + "_".join(
                [f"{k}:{v:.2g}" for k, v in self.fitting_kwargs.items()]) + ".npz")
            if self.fitting_kwargs != loaded["fitting_kwargs"]:
                raise ValueError("Fitting kwargs don't match, refitting")
            self.sim_dynamics = loaded["sim_dynamics"]
            self._fitted_n = loaded["fitted_n"]
            self._fitted_nu = loaded["fitted_nu"]
            self.simulated_dynamics_loss = loaded["simulated_dynamics_loss"]
            return self._fitted_n
        except:
            pass
        np.random.seed(seed)
        if max_n is None:
            max_n = self.fitting_kwargs['base_n'] * 0.95
        self._fitted_n = np.full(self.dynamics.shape[:-1], np.nan)
        self._fitted_nu = np.full(self.dynamics.shape[:-1], np.nan)
        self.simulated_dynamics_loss = np.full(self.dynamics.shape[:-1], np.nan)
        increased_freq_factor = self.fitting_kwargs.get('fit_scale_factor', 50)
        new_x = np.linspace(-self.pre_lag, self.post_lag, ((self.dynamics.shape[-1] - 1) * increased_freq_factor) + 1)
        true_freq = np.full_like(new_x, np.nan)
        np.random.seed(97)
        for block in range(2, self.dynamics.shape[0]):
            for direction in range(2):
                if np.isnan(self.dynamics[block, direction]).all():
                    continue

                pre_change = 500 + (((-1) ** direction) * (self.diffs[block] / 2))
                post_change = 500 + (((-1) ** (direction + 1)) * (self.diffs[block] / 2))
                comparator = np.greater if direction == 0 else np.less
                # if there was no update
                if comparator(self.dynamics[block, direction, self.pre_lag + 1:], 500).all():
                    continue
                # if the update happened before the change
                if (~comparator(self.dynamics[block, direction, :self.pre_lag + 1], 500)).any():
                    continue
                true_freq[:(increased_freq_factor * self.pre_lag + 1)] = pre_change
                true_freq[(increased_freq_factor * self.pre_lag + 1):] = post_change
                spl = Akima1DInterpolator(self.lags, self.dynamics[block, direction])
                interpolated_dynamics = spl(new_x)
                signal = np.repeat(true_freq, self.fitting_kwargs["n_neurons"]).reshape(
                    (new_x.size, -1))
                # signal += np.random.normal(0, np.sqrt(np.nanmean(self.group_dynamics_std[block, direction])), signal.shape)
                try:
                    params, loss = coarse_to_fine_search(signal, interpolated_dynamics,
                                                         {"n": [min_n, max_n],
                                                          "nu": nu_range},
                                                         self._kalman_filter, self._simple_loss, iterations=10,
                                                         block=block, **self.fitting_kwargs, rtol=0, atol=0,
                                                         post_lag=7, pre_lag=2, tqdm=False, seed=97)
                except ValueError:
                    continue
                self.simulated_dynamics_loss[block, direction] = loss
                self._fitted_n[block, direction] = params["n"]
                self._fitted_nu[block, direction] = params["nu"]
                self.sim_dynamics[block, direction] = \
                    self._kalman_filter(signal, params["n"], nu=params["nu"], seed=seed, **self.fitting_kwargs)[
                        [(i * increased_freq_factor) for i in range(self.dynamics.shape[-1])]]
        np.savez(f"data/{self.name}_ei_fits/{self.name}{str(self.subject_idx).zfill(3)}_" + "_".join(
            [f"{k}:{v:.2g}" for k, v in self.fitting_kwargs.items()]) + ".npz",
                 sim_dynamics=self.sim_dynamics, fitted_n=self._fitted_n, fitted_nu=self._fitted_nu,
                 simulated_dynamics_loss=self.simulated_dynamics_loss, fitting_kwargs=self.fitting_kwargs)
        return self._fitted_n

    def fit_n(self):
        return np.nanmean(self.fit_n_per_block())

    def get_change_dynamics(self, pre_lag=2, post_lag=7, data=None):
        if data is None:
            data = self.d
        window_size = pre_lag + post_lag + 1
        change_dynamics = np.zeros((self.nblocks - 1, self.nreps, 2, window_size))
        freq_diffs = []
        for i in range(1, self.nblocks):
            freq_diffs.append(np.abs(np.diff(np.unique(self.s[i, 0][~np.isnan(self.s[i, 0])]))).item())
            for j in range(self.nreps):
                change_dynamics[i - 1, j, ...] = self._single_block_dynamics(data[i, j], i, j, pre_lag, post_lag)
        return np.nanmean(change_dynamics, axis=1), np.array(freq_diffs), np.arange(-pre_lag, post_lag + 1)

    def _single_block_dynamics(self, data, block, rep, pre_lag, post_lag):
        window_size = pre_lag + post_lag + 1
        acc_idx = np.where(np.diff(self.s[block, rep]) < 0)[0] + 1
        dec_idx = np.where(np.diff(self.s[block, rep]) > 0)[0] + 1
        change_dynamics = np.full((2, window_size), np.nan)
        for k, change_idx in enumerate([acc_idx, dec_idx]):
            if change_idx[0] - pre_lag < 0:
                change_idx = change_idx[1:]
            over_time_idx = np.argmax(change_idx + post_lag >= self.ntimes)
            if over_time_idx > 0:
                change_idx = change_idx[:over_time_idx]
            indices = [list(range(ch - pre_lag, ch + post_lag + 1)) for ch in change_idx]
            indices = [item for sublist in indices for item in sublist]
            change_dyn = data[indices].reshape((change_idx.size, window_size))
            change_dyn = change_dyn[~np.isnan(change_dyn).any(axis=1)]
            if change_dyn.shape[0] > 3:
                change_dynamics[k, :] = np.nanmean(change_dyn, axis=0)
        pre_change_freq = np.array([np.nanmax(self.s[block, rep]), np.nanmin(self.s[block, rep])])
        change_dynamics -= np.nanmean(change_dynamics[..., :pre_lag], axis=-1, keepdims=True) - pre_change_freq[:, None]
        return change_dynamics

    def plot_change_dynamics(self, c=None):
        if c is None:
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        pre_change_metronome = np.stack([np.nanmax(self.s[1:, 0, :], axis=-1), np.nanmin(self.s[1:, 0, :], axis=-1)],
                                        axis=-1)
        dynamics = self.dynamics - np.nanmean(self.dynamics[..., np.array(self.lags) < 0], axis=-1,
                                              keepdims=True) + pre_change_metronome[..., None]
        if np.isnan(self.sim_dynamics).any():
            sim_dynamics = self.sim_dynamics - np.nanmean(self.sim_dynamics[..., np.array(self.lags) < 0], axis=-1,
                                                          keepdims=True) + pre_change_metronome[..., None]
        fig, axes = plt.subplots(nrows=dynamics.shape[0] - 2, ncols=dynamics.shape[1], sharex='all',
                                 sharey='row')
        change_idx = np.where(self.lags == 0)[0][0]
        true_freq = np.zeros_like(self.lags)
        for i, row_axes in enumerate(axes[::-1]):
            i += 2
            row_axes[0].set_ylabel(f"{int(self.diffs[i])}ms step\ndelay interval (ms)")
            for j, ax in enumerate(row_axes):
                true_freq[:change_idx] = 500 + (((-1) ** j) * (self.diffs[i] / 2))
                true_freq[change_idx:] = 500 + (((-1) ** (j + 1)) * (self.diffs[i] / 2))
                ax.plot(self.lags, true_freq, linestyle=':', color='k')
                if np.isnan(self.sim_dynamics).any():
                    ax.plot(self.lags, dynamics[i, j], '-o', color="gray")
                    ax.plot(self.lags, sim_dynamics[i, j], '-o', color=c, label=f"n={self._fitted_n[i, j]:.2g}")
                    ax.legend()
                else:
                    ax.plot(self.lags, dynamics[i, j], '-o', color=c)
                # ax.set_ylim(true_freq.min() - 10, true_freq.max() + 10)
        axes[0, 0].set_title("Accelerating")
        axes[0, 1].set_title("Decelerating")
        return fig


# %%

def plot_change_dynamics_helper(s, group_dynamics, diffs, lags, axes=None, label=None, c=None):
    pre_change_metronome = np.stack([np.nanmax(s[:, 1:, 0, :], axis=-1), np.nanmin(s[:, 1:, 0, :], axis=-1)], axis=-1)
    group_dynamics = np.stack(group_dynamics)
    group_dynamics = group_dynamics - np.nanmean(group_dynamics[..., np.array(lags) < 0], axis=-1,
                                                 keepdims=True) + pre_change_metronome[..., None]
    group_dynamics_mean = np.nanmean(group_dynamics, axis=0)
    group_dynamics_std = np.nanstd(group_dynamics, axis=0)
    if axes is None:
        fig, axes = plt.subplots(nrows=group_dynamics.shape[1] - 2, ncols=group_dynamics.shape[2], sharex='all',
                                 sharey='row')
    change_idx = np.where(lags == 0)[0][0]
    true_freq = np.zeros_like(lags)
    for i, row_axes in enumerate(axes[::-1]):
        i += 2
        row_axes[0].set_ylabel(f"{int(diffs[i])}ms\ndelay interval (ms)")
        for j, ax in enumerate(row_axes):
            if c is not None:
                ax.errorbar(lags, group_dynamics_mean[i, j], group_dynamics_std[i, j], None, '-o', ecolor="k",
                            label=label, color=c)
            else:
                ax.errorbar(lags, group_dynamics_mean[i, j], group_dynamics_std[i, j], None, '-o', ecolor="k",
                            label=label)
            true_freq[:change_idx] = 500 + (((-1) ** j) * (diffs[i] / 2))
            true_freq[change_idx:] = 500 + (((-1) ** (j + 1)) * (diffs[i] / 2))
            ax.plot(lags, true_freq, linestyle=':', color='k')
            ax.set_ylim(true_freq.min() - 10, true_freq.max() + 10)
    axes[0, 0].set_title("Accelerating")
    axes[0, 1].set_title("Decelerating")


def plot_response_vs_fit(data, title=None, c_model=None, fig=None, axes=None):
    subj1_dyn, diffs, lags = data.subjects[0].dynamics, data.subjects[0].diffs, data.subjects[0].lags
    real_change_dyn = [subj.dynamics for subj in data.subjects]
    est_change_dyn = [subj.sim_dynamics for subj in data.subjects]
    if fig is None:
        fig, axes = plt.subplots(nrows=subj1_dyn.shape[0] - 2, ncols=subj1_dyn.shape[1], sharex='all', sharey='row',
                                 figsize=(5 * subj1_dyn.shape[1], 2 * subj1_dyn.shape[0]))
    plot_change_dynamics_helper(data.s, real_change_dyn, diffs, lags, axes, label="real", c="gray")
    plot_change_dynamics_helper(data.s, est_change_dyn, diffs, lags, axes, label="model", c=c_model)
    for ax in axes.ravel():
        ax.legend()
    if title is not None:
        fig.suptitle(title)
    return fig, axes


def plot_full_data_response_vs_fit(data: TappingData, title=None, c_model=None, fig=None, axes=None, ylabel=True):
    if fig is None:
        fig, axes = plt.subplots(nrows=data.group_dynamics_mean.shape[0] - 2, ncols=data.group_dynamics_mean.shape[1],
                                 sharex='all', sharey='row', figsize=(5 * data.subjects[0].dynamics.shape[1],
                                                                      2 * data.subjects[0].dynamics.shape[0]))
    change_idx = np.where(data.lags == 0)[0][0]
    true_freq = np.zeros_like(data.lags)
    for i, row_axes in enumerate(axes[::-1]):
        i += 2
        if ylabel:
            if i==3:
                row_axes[0].set_ylabel(f"{int(data.diffs[i])}ms\ndelay interval (ms)")
            else:
                row_axes[0].set_ylabel(f"{int(data.diffs[i])}ms\n")
        for j, ax in enumerate(row_axes):
            if c_model is not None:
                ax.errorbar(data.lags, data.group_dynamics_mean[i, j], data.group_dynamics_std[i, j], None, '-o',
                            ecolor="k", label="real", color="gray")
                ax.plot(data.lags, data.simulated_dynamics_mean[i, j], '-o', color=c_model, label="model")
            else:
                ax.errorbar(data.lags, data.group_dynamics_mean[i, j], data.group_dynamics_std[i, j], None, '-o',
                            ecolor="k", label="real")
                ax.plot(data.lags, data.simulated_dynamics_mean[i, j], '-o', label="model")

            true_freq[:change_idx] = 500 + (((-1) ** j) * (data.diffs[i] / 2))
            true_freq[change_idx:] = 500 + (((-1) ** (j + 1)) * (data.diffs[i] / 2))
            ax.plot(data.lags, true_freq, linestyle=':', color='k')
            ax.set_ylim(true_freq.min() - 10, true_freq.max() + 10)
    axes[0, 0].set_title("Accelerating")
    axes[0, 1].set_title("Decelerating")
    fig.text(0.29, 0.04, 'Lags', ha='center', fontweight="bold", fontsize=20)
    fig.text(0.75, 0.04, 'Lags', ha='center', fontweight="bold", fontsize=20)
    axes[0, 1].legend(bbox_to_anchor=[-0.45, 0.3], loc="lower left")
    axes[-1, 0].set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    axes[-1, 1].set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    if title is not None:
        fig.suptitle(title)
    return


def permutation_fit_to_group_dynamics(asd_data, nt_data, n_permutations=1000):
    permuted_fitted_n = np.full((n_permutations,) + asd_data.fitted_n.shape, np.nan)
    for i in tqdm(range(n_permutations), desc="Permutation test"):
        e = np.random.permutation(np.vstack([asd_data.e, nt_data.e]))
        r = np.random.permutation(np.vstack([asd_data.r, nt_data.r]))
        s = np.random.permutation(np.vstack([asd_data.s, nt_data.s]))
        d = np.random.permutation(np.vstack([asd_data.d, nt_data.d]))
        perm_asd_data = TappingData(None, e=e[:len(asd_data)], r=r[:len(asd_data)], s=s[:len(asd_data)],
                                    d=d[:len(asd_data)], **asd_data.fitting_kwargs)
        perm_nt_data = TappingData(None, e=e[len(asd_data):], r=r[len(asd_data):], s=s[len(asd_data):],
                                   d=d[len(asd_data):], **nt_data.fitting_kwargs)
        perm_asd_data.fit_to_group_dynamics(iterations=3)
        perm_nt_data.fit_to_group_dynamics(iterations=3)
        permuted_fitted_n[i] = np.nanmean(perm_nt_data.fitted_n) - np.nanmean(perm_asd_data.fitted_n)
    return permuted_fitted_n


# %%


