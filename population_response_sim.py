# %% imports
import numpy as np
import constants as c
import idr_utils as utils
import plotting as pl


# %% Population response
def simulate_population_response(si=False):
    print("========================================\n"
          "==== Population response simulation ====\n"
          "========================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.PR_NUM_S_LEVELS)
    population_s = np.repeat(s, c.PR_N_NEURONS).reshape((c.PR_N_NEURONS,) + s.shape, order='F')
    print("Calculating neurons gain functions...")

    if si:
        gain_func = utils.logistic_func
        n = c.SI_PR_SLOPE
    else:
        gain_func = utils.hill_func
        n = c.PR_N

    nt_km = c.PR_KM + c.PR_NT_K_STD * np.random.uniform(-1, 1, size=(c.PR_N_NEURONS, 1))
    nt_resp = gain_func(population_s, n, nt_km)
    asd_km = c.PR_KM + c.PR_ASD_K_STD * np.random.uniform(-1, 1, size=(c.PR_N_NEURONS, 1))
    asd_resp = gain_func(population_s, n, asd_km)

    save_prefix = "logistic_func/" if si else ""
    print("Plotting population response...")
    fig_population_resp = pl.plot_population_response(s, asd_resp, nt_resp)
    pl.savefig(fig_population_resp, save_prefix + "NT vs ASD population response", shift_x=0, shift_y=1.01, tight=False,
               si=si)

    effective_n_asd = utils.get_effective_n(asd_resp.mean(0), np.squeeze(s), c.PR_KM)
    effective_n_nt = utils.get_effective_n(nt_resp.mean(0), np.squeeze(s), c.PR_KM)
    print("ASD effective n=%.2f, NT effective n =%.2f" % (effective_n_asd, effective_n_nt))
    asd_min_sig = s[np.argmax(asd_resp.mean(0) >= 0.1)]
    asd_max_sig = s[np.argmax(asd_resp.mean(0) >= 0.9)]
    nt_min_sig = s[np.argmax(nt_resp.mean(0) > 0.1)]
    nt_max_sig = s[np.argmax(nt_resp.mean(0) > 0.9)]
    print("ASD dynamic range: [%.4f,%.4f]\nNT dynamic range: [%.4f,%.4f],R_ASD: %.2f, R_NT:%.2f" % (
        asd_min_sig, asd_max_sig, nt_min_sig, nt_max_sig, asd_max_sig / asd_min_sig, nt_max_sig / nt_min_sig))


if __name__ == '__main__':
    simulate_population_response()
