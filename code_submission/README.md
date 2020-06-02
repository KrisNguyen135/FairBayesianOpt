# Scarce Societal Resource Allocation and the Price of (Local) Justice

This repository is the official implementation of [Scarce Societal Resource Allocation and the Price of (Local) Justice]().

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

[Gurobi](https://www.gurobi.com) (via the [PuLP](https://pypi.org/project/PuLP/) interface) is the IP solver we used to optimize for the leximin objective under cost matrices that are not row-sorted. More information and installation instructions for Gurobi can be found on its homepage: [https://www.gurobi.com](https://www.gurobi.com).

## Running

The results and visualizations presented in our paper can be reproduced by running the cells in the Jupyter notebooks stored in the `notebooks` folder. Specifically:
- The `Empirical PoF and bounds.ipynb` file contains code to reproduce Figures 2 and 3 in the paper.
- The `Average PoF heatmap.ipynb` file contains code to reproduce Figure 4 in the paper and Figure 1 in the appendix.

## Saved simulation results

Most of the experiments with synthetic data in our work can take significant time to run. We thus provide saved data (stored in the `notebooks/saved_sim_results` folder), which is used by the Jupyter notebooks by default. To rerun all the simulations in each notebook, change the `rerun` Boolean variable to `True` before running the cells.
