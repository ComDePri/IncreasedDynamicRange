import numpy as np
import constants as c
import idr_utils as utils
import plotting as pl


def simulate_signal_differences(si=False):
    print("======================================\n"
          "========= Signal differences ========\n"
          "======================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.SSD_NUM_S_LEVELS)
    if si:
        gain_func = utils.logistic_func
        asd_n = c.SI_SSD_ASD_N
        nt_n = c.SI_SSD_NT_N
    else:
        gain_func = utils.ei_gain_func
        asd_n = c.SSD_ASD_N
        nt_n = c.SSD_NT_N
    nt_resp = gain_func(s, nt_n, c.SSD_KM)
    asd_resp = gain_func(s, asd_n, c.SSD_KM)
    ei_resp = gain_func(s, nt_n, c.SSD_KM, c.EI_NU)
    nt_dr = [s[np.argmin(np.abs(0.1 - nt_resp))], s[np.argmin(np.abs(0.9 - nt_resp))]]
    asd_dr = [s[np.argmin(np.abs(0.1 - asd_resp))], s[np.argmin(np.abs(0.9 - asd_resp))]]
    ei_dr = [s[np.argmin(np.abs(0.1 * ei_resp.max() - ei_resp))], s[np.argmin(np.abs(0.9 * ei_resp.max() - ei_resp))]]
    print(f"NT dynamic range: {np.round(nt_dr, 2)}, R={nt_dr[1] / nt_dr[0]:.2f}")
    print(f"ASD dynamic range: {np.round(asd_dr, 2)}, R={asd_dr[1] / asd_dr[0]:.2f}")
    print(f"E\I dynamic range: {np.round(ei_dr, 2)}, R={ei_dr[1] / ei_dr[0]:.2f}")
    signal_sensitivity_fig = pl.plot_sensitivity_to_signal_differences(s, nt_resp, asd_resp, None, si=si)
    savename_prefix = "logistic_func/" if si else ""
    pl.savefig(signal_sensitivity_fig, savename_prefix + "sensitivity to signal differences",
               ignore=signal_sensitivity_fig.get_axes(), tight=False, si=si)
    signal_sensitivity_fig = pl.plot_sensitivity_to_signal_differences(s, nt_resp, asd_resp, ei_resp, si=si)
    pl.savefig(signal_sensitivity_fig, savename_prefix + "sensitivity to signal differences",
               ignore=signal_sensitivity_fig.get_axes(), tight=False, si=True)


if __name__ == '__main__':
    simulate_signal_differences()
