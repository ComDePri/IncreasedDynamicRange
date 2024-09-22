# %% imports
import belief_updating_sim
import binocular_rivalry
import comparisons
import constants as c
import encoding_capacity_sim
import idr_utils as utils
import lca_simulations
import plotting as pl
import population_response_sim
import separatrix_sim
import signal_differences_sim
import sim_power_analysis
import variance_over_signal_range_sim

utils.reload(c)
utils.reload(pl)
utils.make_fig_dir()
# %% simulations
if __name__ == '__main__':
    signal_differences_sim.simulate_signal_differences()  # Sensitivity to signal differences

    variance_over_signal_range_sim.variance_simulation()  # Variance over signal range - E/I VS IDR

    belief_updating_sim.simulate_signal_change_tracking_update()  # Slower responses to sharp transitions using kalman filter

    binocular_rivalry.simulate_binocular_rivalry()  # binocluar rivalry - noisy signal around 0.5

    lca_simulations.simulate_robertson()

    encoding_capacity_sim.simulate_encoding_capacity()  # FI based encoding capacity

    population_response_sim.simulate_population_response()  # Population response

    separatrix_sim.simulate_separatrix()  # separatrix

    sim_power_analysis.simulate_power_analysis() # Power analysis

    comparisons.simulate_comparisons() # HGF and rosenberg comparison

    pl.close_all()
