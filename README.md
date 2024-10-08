
# Increased Dynamic Range: Analysis, Simulation, and Plotting Code Repository

This repository contains all the analysis, simulation, and plotting code used in the scientific publication "Autism spectrum disorder variation as a computational trade-off via dynamic range of neuronal population responses". 
Each script corresponds to a specific figure or set of analyses presented in the paper.

## Repository Structure

- **data/**: Contains the data generated by long simulations and essential images required for figure reproduction.
  - `data_x_y/`: Directories holding the simulation data for different conditions (`x_y` indicates the specific simulation setting).
  - `hgf.mat`: Hierarchical Gaussian Filter (HGF) model simulation output.
  - `stim_image.png`: Stimulus image used in the simulations.
  - `Encoding figure.png`: Example encoding figure referenced in the publication.
  - `button press.png`: Illustration of button press responses.
  - `tapping/`: Directory that should contain the tapping data files for the `belief_updating_sim.py` script, see below.
  - `encoding`: Directory containing the total FI data used in the `encoding_capacity_sim.py` script. This data was generated from [The Github repository of the original publication](https://github.com/cpc-lab-stocker/ASD_Encoding_2020), from the fig3_combined.m script `allTotal_woFB` variables for ASd and NT groups.  

- **figures/**: Contains the figures generated by the scripts in the repository.
- **SI figures/**: Supplementary figures generated by the `SI.py` script and other simulation scripts.
- **constants.py**: Defines constant values used throughout the simulations and analyses, such as parameter values and model settings.
- **idr_utils.py**: Utility functions for data manipulation, statistical analysis, and other auxiliary functions used in different scripts.
- **plotting.py**: Contains various plotting functions used across the different analysis scripts. This file abstracts the plotting logic, allowing for consistent figure styling and formatting.
- **data_fitting.py**: Fits the model to the empirical data and plots the fitting results.


## Figure recreation & simulations
- **main.py**: Main script that sequentially runs all the simulations and analysis as presented in the publication.
- **signal_differences_sim.py**: Recreates Figure 1.
- **variance_over_signal_range_sim.py**: Recreates Figure 2.
- **belief_updating_sim.py**: Recreates Figure 3. For this file, you need 2 .mat files in `data/tapping`. These files can be created from [Vishne et al. publication data](https://osf.io/83wnu/) using the `save_tapping_data.m` script.  
- **binocular_rivalry.py**: Recreates Figure 4.
- **LCA.py & lca_simulations.py**: Recreates Figure 5.
- **encoding_capacity_sim.py**: Recreates Figure 6.
- **population_response_sim.py**: Recreates Figure 7.
- **separatrix_sim.py**: Recreates Figure 8.
- **sim_power_analysis.py**: Recreates Extended Data Figure 1.
- **comparisons.py**: Recreates Extended Data Figure 2.
- **SI.py**: Generates supplementary text figures.
- **increased_dynamic_range_hgf.m**: MATLAB script for simulating the Hierarchical Gaussian Filter (HGF) model, creates the data of `data/hgf.mat`.
- **save_tapping_data.m**: MATLAB script for saving the tapping data in the correct format for the `belief_updating_sim.py` script.
- **total FI derivation.nb**: Notebook file containing the derivation of Fisher Information for the model, used for Figure 6.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries: `numpy`, `scipy`, `matplotlib`, `pandas`
- MATLAB (for running `increased_dynamic_range_hgf.m`, `save_tapping_data.m`)
- Mathematica (for viewing and executing `.nb` files)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ComDePri/IncreasedDynamicRange.git
   cd IncreasedDynamicRange
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. For MATLAB scripts, ensure you have the necessary toolboxes installed.

### Running the Scripts

1. **Main Analysis and Simulations**: To run all the simulations and generate the main figures in the order presented in the publication, execute:
   ```bash
   python main.py
   ```

2. **Supplementary Text Figures**: To generate the supplementary figures, run:
   ```bash
   python SI.py
   ```

3. **Individual Scripts**: You can run individual scripts to reproduce specific figures:
   ```bash
   python belief_updating_sim.py
   ```

### Data

The `data/` directory contains pre-generated data from long simulations. If you want to regenerate the data, please ensure you have sufficient computational resources as these simulations can be time-consuming.

### Plotting and Figure Customization

The `plotting.py` script contains all the necessary functions for generating and customizing figures. You can modify the functions here if you need to adjust figure properties such as labels, colors, or formats.

## Contact

For any questions or inquiries, please contact `oded.wertheimer@mail.huji.ac.il`.
