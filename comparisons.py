import plotting as pl
import idr_utils as utils

import lca_simulations
import encoding_capacity_sim

def simulate_comparisons():
    asd_fit, sim_fit, alphas, omegas = utils.load_hgf()
    fig = pl.plt.figure(figsize=pl.get_fig_size(0.75, 2))
    gs = pl.GridSpec(1, 4, figure=fig, width_ratios=[1.3,0.05, 2, 2])
    axes = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[0,3])]
    pl.plot_hgf_boxplots(omegas, axes[0], "Volatility estimates", r"$\omega_3$")
    encoding_capacity_sim.rosenberg_simulation(axes[1], title="Encoding capacity")
    lca_simulations.simulate_rosenberg(axes[2], title="Motion coherence")
    pl.savefig(fig, "hgf and rosenberg comparison.tiff", tight=True, shift_y=1.2, shift_x=-.15,dpi=96)
    pl.close_all()
# %%
if __name__ == '__main__':
    simulate_comparisons()
