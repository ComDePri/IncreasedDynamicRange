from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wald, invgauss
from sde_simulator import SDESimulatorBase


def relu(x):
    return np.maximum(0., x)


class LCA(SDESimulatorBase):
    def __init__(self, leak, inhibition, I, initial_condition_noise=0., gain_func=lambda x: x, **kwargs):
        super(LCA, self).__init__(y0=np.array([0.5, 0.5]), **kwargs)
        self.initial_condition_noise = initial_condition_noise
        self.y[0] = np.random.normal(0, initial_condition_noise, self.y.shape[1:])
        self.leak = leak
        self.inhibition = inhibition
        self._I = np.array(I)[:, None]
        self._gain_func = gain_func
        self.I = self._gain_func(self._I)
        self.I[0] = 1 - self.I[1]
        self.noise_level = self.noise
        self._rt_stats = None

    @property
    def rt_stats(self):
        if self._rt_stats is None:
            self.get_response_stats(threshold=1.)
        return self._rt_stats

    def deterministic_func(self, y, i, **kwargs):
        return self.I - self.leak * y - self.inhibition * np.flip(y, 0)

    def _step(self, i, **kwargs):
        super()._step(i, **kwargs)
        self.y[i, ...] = relu(self.y[i, ...])

    def stochastic_func(self, y, i, **kwargs):
        return self.noise_level

    def plot_trajectory(self, trajectory_idx, threshold=1., labels=None, ax=None, colors=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax: plt.Axes
        legend = True
        if labels is None:
            labels = ["left", "right"]
        elif not labels:
            labels = ["", ""]
            legend = False
        max_time = np.max(np.argmax(self.y[:, :, trajectory_idx] > threshold, axis=0))
        if max_time + 20 < self.y.shape[0]:
            max_time += 20
        if colors is None:
            ax.plot(self.time[:max_time], self.y[:max_time, 0, trajectory_idx], label=labels[0])
            ax.plot(self.time[:max_time], self.y[:max_time, 1, trajectory_idx], label=labels[1])
        else:
            ax.plot(self.time[:max_time], self.y[:max_time, 0, trajectory_idx], label=labels[0], color=colors[0])
            ax.plot(self.time[:max_time], self.y[:max_time, 1, trajectory_idx], label=labels[1], color=colors[1])
        if legend:
            ax.legend()
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=3)
        ax.axhline(-threshold, color="gray", linestyle=":", linewidth=3)
        ax.set(xlabel="Time")
        return ax.get_figure()

    def plot_mean_trajectory(self, threshold=1.):
        fig, ax = plt.subplots()
        ax: plt.Axes
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=3)
        ax.axhline(-threshold, color="gray", linestyle=":", linewidth=3)
        is_correct_bool_vec, correct_y, pass_times = self._get_correct_y_and_sim_idx(threshold)
        is_incorrect_bool_vec = (~is_correct_bool_vec) & (pass_times[1 - correct_y, :] < self.time.size)

        mean_correct = self.y[:, correct_y, is_correct_bool_vec].mean(-1)
        mean_incorrect = self.y[:, 1 - correct_y, is_incorrect_bool_vec].mean(-1)
        mean_correct_pass = np.argmax(mean_correct > threshold)
        mean_incorrect_pass = np.argmax(mean_incorrect > threshold)

        ax.plot(self.time[:mean_correct_pass], mean_correct[:mean_correct_pass], color="green", label="correct")
        ax.plot(self.time[:mean_incorrect_pass], mean_incorrect[:mean_incorrect_pass], color="red", label="incorrect")

        ax.legend()
        return fig

    def plot_rt(self, threshold=1., ax=None):
        pass_idx = self.get_rt(threshold)
        if ax is None:
            fig, ax = plt.subplots()
            ax: plt.Axes
        rt = self.time[pass_idx]
        self._plot_rt_helper(rt, ax)
        x = np.linspace(0, self.time[-1], 1000)
        if np.array(self.rt_stats["all"]["response_invgauss_fit"]).all():
            ax.plot(x, invgauss.pdf(x, *self.rt_stats["all"]["response_invgauss_fit"]))
        return rt.size

    def _plot_rt_helper(self, pass_time, ax):
        ax.hist(pass_time, bins=50, density=True)
        line_kwargs = {"linestyle": ":", "linewidth": 3}
        if pass_time.size > 0:
            ax.axvline(pass_time.mean(), color="black", label=f"mean:{np.mean(pass_time):.2f}", **line_kwargs)
            ax.axvline(np.median(pass_time), color="red", label=f"median:{np.median(pass_time):.2f}", **line_kwargs)
            ax.legend()

    def get_rt(self, threshold=1.):
        pass_idx = np.argmax(self.y > threshold, 0)
        pass_idx[pass_idx == 0] = self.time.size
        pass_idx = np.min(pass_idx, axis=0)
        pass_idx = pass_idx[pass_idx < self.time.size]
        return pass_idx

    def plot_correct_rt(self, threshold=1, ax=None):
        correct_idx, correct_y_idx, pass_idx = self._get_correct_y_and_sim_idx(threshold)
        pass_idx = pass_idx[correct_y_idx, correct_idx]
        pass_idx = pass_idx[pass_idx < self.time.size]
        rt = self.time[pass_idx]
        if ax is None:
            fig, ax = plt.subplots()
            ax: plt.Axes
        self._plot_rt_helper(rt, ax)
        x = np.linspace(0, self.time[-1], 1000)
        if np.array(self.rt_stats["correct"]["correct_invgauss_fit"]).all():
            ax.plot(x, invgauss.pdf(x, *self.rt_stats["correct"]["correct_invgauss_fit"]))
        return rt.size

    def _get_correct_y_and_sim_idx(self, threshold):
        pass_idx = np.argmax(self.y > threshold, 0)
        pass_idx[pass_idx == 0] = self.time.size
        correct_sign = np.sign(np.diff(np.squeeze(self.I)))[0]
        if correct_sign == 0:
            correct_sign = 1
        correct_y_idx = 0 if correct_sign < 0 else 1
        correct_idx = (((pass_idx[0] - pass_idx[1]) * correct_sign) > 0)
        return correct_idx, correct_y_idx, pass_idx

    def plot_error_rt(self, threshold=1, ax=None):
        correct_idx, correct_y_idx, pass_idx = self._get_correct_y_and_sim_idx(threshold)
        pass_idx = pass_idx[1 - correct_y_idx, ~correct_idx]
        pass_idx = pass_idx[pass_idx < self.time.size]
        rt = self.time[pass_idx]
        if ax is None:
            fig, ax = plt.subplots()
            ax: plt.Axes
        self._plot_rt_helper(rt, ax)
        x = np.linspace(0, self.time[-1], 1000)
        if np.array(self.rt_stats["error"]["error_invgauss_fit"]).all():
            ax.plot(x, invgauss.pdf(x, *self.rt_stats["error"]["error_invgauss_fit"]))
        return rt.size

    def plot_all_rt_dist(self, threshold=1, title="RT"):
        fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharey='all')
        num_responses = self.plot_rt(threshold, axes[0])
        num_correct = self.plot_correct_rt(threshold, axes[1])
        num_error = self.plot_error_rt(threshold, axes[2])
        if num_responses == 0:
            raise Exception()
        n_sims = self.y.shape[-1]
        axes[0].set(title=f"All RT, %Resp={num_responses / n_sims:.2}", xlabel="Time", ylabel="Frequency")
        axes[1].set(
            title=f"Correct RT, %Correct from all={num_correct / n_sims:.2}, "
                  f"%Correct from resp={num_correct / num_responses:.2}", xlabel="Time", ylabel="Frequency")
        axes[2].set(
            title=f"Error RT, %Error from all={num_error / n_sims:.2}, "
                  f"%Error from resp={num_error / num_responses:.2}", xlabel="Time", ylabel="Frequency")
        fig.suptitle(title)

    def get_response_stats(self, threshold=.5):
        pass_idx = self.get_rt(threshold)
        rt = self.time[pass_idx]

        correct_idx, correct_y_idx, pass_idx = self._get_correct_y_and_sim_idx(threshold)
        idx = pass_idx[correct_y_idx, correct_idx]
        idx = idx[idx < self.time.size]
        correct_rt = self.time[idx]

        idx = pass_idx[1 - correct_y_idx, ~correct_idx]
        idx = idx[idx < self.time.size]
        error_rt = self.time[idx]
        rt_mean, rt_median, rt_std, rt_wald_fit_params, percent_rt, rt_invgauss_fit = self.get_rt_stats(rt)

        correct_rt_mean, correct_rt_median, correct_rt_std, correct_rt_wald_fit_params, percent_correct_rt, correct_invgauss = self.get_rt_stats(
            correct_rt)

        error_rt_mean, error_rt_median, error_rt_std, error_rt_wald_fit_params, percent_error_rt, error_invgauss = self.get_rt_stats(
            error_rt)

        all_rt_stats = dict(response_percent=percent_rt, response_mean=rt_mean, response_median=rt_median,
                            response_std=rt_std, response_wald_loc=rt_wald_fit_params[0],
                            response_wald_scale=rt_wald_fit_params[1], response_invgauss_fit=rt_invgauss_fit)
        correct_rt_stats = dict(correct_percent=percent_correct_rt, correct_mean=correct_rt_mean,
                                correct_median=correct_rt_median, correct_std=correct_rt_std,
                                correct_wald_loc=correct_rt_wald_fit_params[0],
                                correct_wald_scale=correct_rt_wald_fit_params[1],
                                correct_invgauss_fit=correct_invgauss)
        error_rt_stats = dict(error_percent=percent_error_rt, error_mean=error_rt_mean,
                              error_median=error_rt_median, error_std=error_rt_std,
                              error_wald_loc=error_rt_wald_fit_params[0], error_wald_scale=error_rt_wald_fit_params[1],
                              error_invgauss_fit=error_invgauss)
        self._rt_stats = {"all": all_rt_stats, "correct": correct_rt_stats, "error": error_rt_stats}
        return all_rt_stats, correct_rt_stats, error_rt_stats

    def get_rt_stats(self, resp_rt):
        if resp_rt.size > 0:
            percent_rt = resp_rt.size / self.y.shape[-1]
            rt_mean = resp_rt.mean()
            rt_median = np.median(resp_rt)
            rt_std = np.std(resp_rt)
            rt_wald_fit_params = wald.fit(resp_rt, floc=0)
            rt_invgauss_fit = invgauss.fit(resp_rt, floc=0)
        else:
            percent_rt = 0
            rt_mean = np.NaN
            rt_median = np.NaN
            rt_std = np.NaN
            rt_wald_fit_params = [None, None]
            rt_invgauss_fit = [None, None, None]

        return rt_mean, rt_median, rt_std, rt_wald_fit_params, percent_rt, rt_invgauss_fit
