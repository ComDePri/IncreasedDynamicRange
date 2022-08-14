# %% imports
import constants as c
import utils
import plotting as pl

from population_response_sim import simulate_population_response
from variance_over_signal_range_sim import variance_simulation
from signal_differences_sim import simulate_signal_differences
from hebbian_learning_sim import simulate_hebbian_learning
from belief_updating_sim import simulate_signal_change_tracking_update
from binocular_rivalry import simulate_binocular_rivalry
from separatrix_sim import simulate_separatrix
from encoding_capacity_sim import simulate_encoding_capacity

utils.reload(c)
utils.reload(pl)
utils.make_fig_dir()
utils.make_si_fig_dir()
# %% simulations

simulate_signal_differences()  # Sensitivity to signal differences

variance_simulation()  # Variance over signal range - E/I VS IDR

simulate_signal_change_tracking_update()  # Slower responses to sharp transitions using kalman filter

simulate_binocular_rivalry()  # binocluar rivalry - noisy signal around 0.5

simulate_hebbian_learning()  # learning rate and accuracy

simulate_encoding_capacity()  # FI based encoding capacity

simulate_population_response()  # Population response

simulate_separatrix()  # separatrix

pl.close_all()
