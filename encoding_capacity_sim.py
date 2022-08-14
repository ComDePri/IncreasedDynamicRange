import numpy as np
import constants as c
import utils
import plotting as pl


def simulate_encoding_capacity():
    print("======================================\n"
          "==== Encoding capacity simulation ====\n"
          "======================================")
    np.random.seed(c.SEED)
    s = np.linspace(0, 1, c.FI_NUM_S)
    fi = []
    for n in c.FI_N_LIST:
        fi.append(utils.population_resp_fi(s, n, c.FI_KM))
    fig_fi = pl.plot_fi(fi, s)
    pl.savefig(fig_fi, "encoding capacity", shift_x=-0.08, shift_y=1.05, tight=False, ignore=fig_fi.get_axes())


if __name__ == '__main__':
    simulate_encoding_capacity()
