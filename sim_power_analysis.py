import numpy as np
import constants as c
import plotting as pl
import idr_utils as utils


def simulate_power_analysis(nt_ns=None, asd_ns=None):
    if nt_ns is None:
        nt_ns = [13, 11]
    elif isinstance(nt_ns, int):
        nt_ns = [nt_ns]
    if asd_ns is None:
        asd_ns = [7, 9]
    elif isinstance(asd_ns, int):
        asd_ns = [asd_ns]
    populations, powers, heterogeneities = [], [], []
    for nt, asd in zip(nt_ns, asd_ns):
        data, individual_ns = utils.load_power_analysis_data(nt, asd)
        population, pow, hets = utils.parse_power_analysis_data(data, individual_ns)
        populations.append(population)
        powers.append(pow)
        heterogeneities.append(hets)

    fig = pl.plot_power_analysis_figure(populations, powers, heterogeneities, nt_ns, asd_ns)
    pl.savefig(fig, "complex_lit.tiff", shift_y=1.05, numbering_size=35,
               figure_coord_labels=[(0.035, 0.925), (0.035, 0.425), (0.525, 0.425)],dpi=96)
    pl.close_all()


if __name__ == '__main__':
    simulate_power_analysis()
