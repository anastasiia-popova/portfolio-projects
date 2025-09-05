# Differentially Private Learning for Blood Cell Image Classification

## Description

This project aims to train an accurate yet compact convolutional neural network with privacy guarantees on the BloodMNIST dataset. We compare the accuracy of the differentially private model to its non-private counterpart to evaluate the privacy-utility trade-off for the compact CNN model trained with DP-SGD. We also investigate the effect of hyperparameter choices on private model performance.

## Environment

The notebooks include instructions for running the code in Google Colab, with support for GPU usage.

To set up the environment locally, use:
`conda env create -f environment.yml`

## How to run and interpret the code 

The main results can be found in the following Jupyter notebooks:

- `demo.ipynb`: comparison of non-private CNN with training using (2, 7e-09)-DP-SGD.
- `parent_non_private_model.ipynb`: GroupNorm layer tuning and learning rate search for DermaMNIST (public), and finding the clipping parameter.
- `non_private_model.ipynb`: training of non-private models (on CPU; similar final data for GPU training is in `/data`).
- `dpsgd_training.ipynb`: training with DP-SGD.

There are also two additional scripts used to generate data:

- `non_private_model.py`: for testing purposes.
- `dpsgd_experiments.py`: to reproduce plots.

Other files:

- `model.py`: contains the compact CNN model and helper functions for training.
- `utility.py`: contains helper functions to simplify code in notebooks and scripts.

Data produced by these scripts can be found in the `/data` folder. Weights obtained from experiments are stored in the `/weights` folder.

Plots are available in the `/plots` folder. The code for generating plots with the obtained data can be found in `plots/summary_plots.ipynb`.

