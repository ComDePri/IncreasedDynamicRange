import numpy as np
import constants as c
import utils
import plotting as pl


def simulate_signal_differences():
    print("======================================\n"
          "========= Signal differences ========\n"
          "======================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.SSD_NUM_S_LEVELS)
    nt_resp = utils.hill_func(s, c.SSD_NT_N, c.SSD_KM)
    asd_resp = utils.hill_func(s, c.SSD_ASD_N, c.SSD_KM)

    signal_sensitivity_fig = pl.plot_sensitivity_to_signal_differences(s, nt_resp, asd_resp)
    pl.savefig(signal_sensitivity_fig, "sensitivity to signal differences", ignore=signal_sensitivity_fig.get_axes(),
               tight=False)


if __name__ == '__main__':
    simulate_signal_differences()
