import numpy as np
import scipy.stats
from tqdm import tqdm

import constants as c
import idr_utils as utils
import plotting as pl
from LCA import LCA


# %%
def get_correct(signal_level, time):
    threshold = 0.51
    asd_params = dict(n=8, c=1.)
    nt_params = dict(n=16, c=1.)
    asd_model, params = simulate_model(asd_params, signal_level, time, threshold)
    nt_model, params = simulate_model(nt_params, signal_level, time, threshold)
    return (asd_model.get_response_stats(threshold)[1]["correct_percent"],
            nt_model.get_response_stats(threshold)[1]["correct_percent"])


def simulate_model(model_params, signal_level, time, threshold, n_sim=200, leak=1, inhibition=0.25, noise=0.32,
                   gain_func_generator=utils.gain_func_generator):
    params = {"max_t": time, "leak": leak, "inhibition": inhibition, "noise": noise,
              "I": [0.5 - signal_level, 0.5 + signal_level], "threshold": threshold, "n_sim": n_sim}
    model = LCA(**params, verbose=False, gain_func=gain_func_generator(**model_params))
    model.simulate(use_tqdm=False)
    return model, params


def lca_power_analysis(het_het=0.1, asd_n=7, nt_n=14):
    np.random.seed(c.SEED)
    MAX_POP = 3000
    N_SIM = 150
    asd_het = utils.get_heterogeneity_from_n(asd_n, 16)
    nt_het = utils.get_heterogeneity_from_n(nt_n, 16)
    asd_rand_het = utils.sample_truncnorm(asd_het, het_het, 0.01, 0.5, MAX_POP)
    nt_rand_het = utils.sample_truncnorm(nt_het, het_het, 0.01, 0.5, MAX_POP)
    asd_ns = utils.get_effective_n_from_heterogeneity(asd_rand_het, 20)
    nt_ns = utils.get_effective_n_from_heterogeneity(nt_rand_het, 20)
    signal_levels = np.linspace(0, 0.06, 31)
    asd_correct_percent = np.zeros((3, MAX_POP, signal_levels.size), dtype=float)
    nt_correct_percent = np.zeros((3, MAX_POP, signal_levels.size), dtype=float)
    times = [2, 15]
    for i, time in enumerate(times):
        for j in tqdm(range(MAX_POP), desc="Individuals"):
            for k, signal_level in enumerate(signal_levels):
                asd_model, params = simulate_model({"n": asd_ns[i], "c": 1.}, signal_level, time, 0.51, 1, N_SIM)
                nt_model, params = simulate_model({"n": nt_ns[i], "c": 1.}, signal_level, time, 0.51, 1, N_SIM)
                asd_correct_percent[i, j, k] = asd_model.get_response_stats(1)[1]["correct_percent"]
                nt_correct_percent[i, j, k] = nt_model.get_response_stats(1)[1]["correct_percent"]
    np.savez(f"data_{nt_n}_{asd_n}/lca_{het_het}.npz", asd=asd_ns, nt=nt_ns)

    asd_threshold_idx = np.argmin(np.abs(asd_correct_percent - 0.8), axis=-1)
    asd_threshold_levels = signal_levels[asd_threshold_idx] / 0.5
    nt_threshold_idx = np.argmin(np.abs(nt_correct_percent - 0.8), axis=-1)
    nt_threshold_levels = signal_levels[nt_threshold_idx] / 0.5

    fig, axes = pl.plt.subplots(1, 3, figsize=pl.get_fig_size(1, 3))
    populations, powers = [], []
    for i in range(3):
        pops, power = utils.power_analysis(asd_threshold_levels[i], nt_threshold_levels[i], scipy.stats.mannwhitneyu,
                                           np.unique(np.round(np.logspace(np.log10(5), np.log10(MAX_POP), 50))))
        pl.plot_power_analysis(pops, power, f"LCA threshold {times[i]}", ax=axes[i])
        populations.append(pops)
        powers.append(power)
    pl.savefig(fig, f"LCA Power analysis for het_het-{het_het} nt n-{nt_n} asd n-{asd_n}")
    return populations, powers


def simulate_robertson():
    asd_correct_percent, ei_correct_percent, nt_correct_percent, signal_levels, times = lca_robertson_simulation()
    params = dict(n=16, c=1.)
    np.random.seed(c.SEED)
    model, params = simulate_model(params, 0.05, 10, 0.51, 200, 1, 0.25, .32)

    fig, ignore = pl.plot_robertson(signal_levels, times, asd_correct_percent, nt_correct_percent, model, 0.158)
    pl.savefig(fig, "Robertson LCA fit", ignore=ignore, tight=False,
               figure_coord_labels=[(0.025, 0.95), (0.025, 0.4), (0.525, 0.4)])

    fig, (ax, ax_robertson) = pl.plt.subplots(1, 2, figsize=pl.get_fig_size(1, 2))
    pl.plot_robertson_vs_lca(ax, ax_robertson, np.argmax(nt_correct_percent > 0.8, axis=1),
                             np.argmax(ei_correct_percent > 0.8, axis=1), signal_levels, times, 0.158,
                             labels=["Full inhibition", "Decreased inhibition"])
    pl.savefig(fig, "Robertson vs LCA EI", si=True)
    pl.close_all()


def lca_robertson_simulation(threshold=0.51, n_sim=200, leak=1, inhibition=0.25, noise=0.32):
    times = np.array([2, 4, 15])
    signal_levels = np.linspace(0, 0.06, 31)
    params = [dict(n=8, c=1.),
              dict(n=16, c=1.),
              dict(n=16, c=.75)]
    correct = []
    for p in params:
        correct.append([])
        for time in times:
            correct[-1].append([])
            for sig in signal_levels:
                np.random.seed(c.SEED)
                model, _ = simulate_model(p, sig, time, threshold, n_sim, leak, inhibition, noise)
                correct[-1][-1].append(model.get_response_stats(threshold)[1]["correct_percent"])
    asd_correct_percent, nt_correct_percent, ei_correct_percent = map(np.array, correct)
    return asd_correct_percent, ei_correct_percent, nt_correct_percent, signal_levels, times


def simulate_rosenberg(ax1=None, title=None, threshold=0.6, n_sim=200, leak=.51, inhibition=0.2, noise=0.3):
    times = np.array([2, 4, 15])
    signal_levels = np.linspace(0, 0.35, 31)
    params = [dict(c=.75),
              dict(c=1.)]
    correct = []
    for p in params:
        correct.append([])
        for time in times:
            correct[-1].append([])
            for sig in signal_levels:
                np.random.seed(c.SEED)
                model, _ = simulate_model(p, sig, time, threshold, n_sim, leak, inhibition, noise,
                                          utils.rosenberg_gain_func_generator)
                correct[-1][-1].append(model.get_response_stats(threshold)[1]["correct_percent"])
    asd_correct_percent, nt_correct_percent = map(np.array, correct)

    if ax1 is None:
        fig, (ax, ax_robertson) = pl.plt.subplots(1, 2, figsize=pl.get_fig_size(1, 2))
    else:
        ax, ax_robertson = ax1, None
        fig = None
    nt_threshold_idx = np.argmax(nt_correct_percent > 0.8, axis=1)
    nt_threshold_idx[nt_correct_percent.max(axis=1) < 0.8] = signal_levels.size - 1
    asd_threshold_idx = np.argmax(asd_correct_percent > 0.8, axis=1)
    asd_threshold_idx[asd_correct_percent.max(axis=1) < 0.8] = signal_levels.size - 1
    pl.plot_robertson_vs_lca(ax, ax_robertson, nt_threshold_idx, asd_threshold_idx, signal_levels, times, 0.5,
                             labels=["Full inhibition", "Decreased inhibition"])
    # ax.set_ylim(0, 100)
    if title is not None:
        ax.set_title(title)
    if fig is not None:
        pl.savefig(fig, "Rosenberg robertson vs LCA", si=True, tight=True, shift_y=1.05)
        pl.close_all()
    return asd_correct_percent, nt_correct_percent, signal_levels, times


if __name__ == '__main__':
    simulate_robertson()

