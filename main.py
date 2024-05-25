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
import variance_over_signal_range_sim
import sim_power_analysis

utils.reload(c)
utils.reload(pl)
utils.make_fig_dir()
# %% simulations
if __name__ == '__main__':
    signal_differences_sim.simulate_signal_differences()  # Sensitivity to signal differences
    signal_differences_sim.simulate_rosenberg()  # Sensitivity to signal differences

    variance_over_signal_range_sim.variance_simulation()  # Variance over signal range - E/I VS IDR
    variance_over_signal_range_sim.rosenberg_simulation()  # Variance over signal range - E/I VS IDR

    belief_updating_sim.simulate_signal_change_tracking_update()  # Slower responses to sharp transitions using kalman filter
    belief_updating_sim.rosenberg_simulation()  # Slower responses to sharp transitions using kalman filter

    binocular_rivalry.simulate_binocular_rivalry()  # binocluar rivalry - noisy signal around 0.5
    binocular_rivalry.rosenberg_simulation()  # binocluar rivalry - noisy signal around 0.5

    encoding_capacity_sim.simulate_encoding_capacity()  # FI based encoding capacity
    encoding_capacity_sim.rosenberg_simulation()  # FI based encoding capacity

    population_response_sim.simulate_population_response()  # Population response

    separatrix_sim.simulate_separatrix()  # separatrix

    lca_simulations.simulate_robertson()
    lca_simulations.simulate_rosenberg()

    comparisons.simulate_comparisons()

    sim_power_analysis.simulate_power_analysis()

    pl.close_all()
