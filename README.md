# IncreasedDynamicRange
Code for *Increased dynamic range as a driver of ASD behavioral differences*.

# Recreating figures
In order to run the simulations and create and save the figures, the working directory should contain directories named "figures" and "SI figures". They can be created by running the `make_fig_dir` and `make_si_fig_dir` functions defined in `utils.py`
Every figure in the main paper has a corresponding .py file running the simulation and plotting the figures. Some files create supplementary figures as well.
Each file can be run alone, or imported and run.

The file main.py runs the simulations in the order they are presented in the paper and provides and example of importing and running the simulations from a separate file.

# Supplementary information
Most of the supplementary figures and information are generated in the `SI.py` file.
The file is separated into logical blocks with the `# %%` marking, with an informative header.

# Helper files
* **`plotting.py`**: This files sets plotting constants, defines helper functions for plotting and specific function for plots
* **`utils.py`**: This file contains utility functions performing various calculations that are used in the simulations. 
* **`contants.py`**: This files contains the constants used in all files - plotting parameters and simulation constants. Changing values in this files changes the parameters of the correspinding simulation.

# HGF simulation
To recreate the HGF figures with the provided `hgf.mat` data, run the last section of `SI.py`.
In order to run the HGF simulation from scratch, you must download the [tapas toolbox](https://github.com/translationalneuromodeling/tapas) and add it to you matlab path. Then run the `increased_dynamic_range_hgf.m` to save the hgf.mat file and run the last section of `SI.py` to generate the figures.