<!-- >📋  A template README.md for code accompanying a Machine Learning paper -->
# My Paper Title

This Github repository is part of our submission to the 2021 MLRC Reproducibility Challenge, where we attempt to reproduce the results and verify the claims of [Identifying through Flows for Recovering Latent Representations](https://arxiv.org/abs/1909.12555). This repository is a modified version of the official [GitHub implementation](https://github.com/MathsXDC/iFlow). Futhermore, the added RealNVP implementation for visualizing the difference between iFlow and Flow is adapted from this [GitHub RealNVP implementation](https://github.com/senya-ashukha/real-nvp-pytorch.)


## Requirements

Python>=3.6.8.

To install requirements for training on the GPU:

```setup
pip install -r requirements_GPU.txt
```

And for training on the CPU:

```setup
pip install -r requirements_CPU.txt
```

Note here that Python 3.9 users should add ```-c=conda-forge``` to the installation of Pytorch.

## Training

In our research numerous configurations of the model has been trained and evaluated. These models were trained across two different datasets: the original TLC dataset and the half-moon dataset.

### TCL Dataset

To train the different models on the TCL dataset, code/main.py needs to be executed. In this file there are several important parameters:

* -i Determines the network to use. Can either be iFlow (default) or iVAE
* -x Contains the arguments for data generation in the format: nps_ns_dl_dd_nl_s_p_a_u_n, where:
    + nps: number of points per segment
    + ns: number of segments
    + dl: dimension of latent space
    + dd: dimension of data space
    + nl: number of layers to generate data with
    + s: random seed. Can be an integer, or n for a randomized seed
    + p: prior distribution. Can be gauss, hs (Hypersecant), lap (laplacian) or mixture
    + a: activation function. Can be none, lrelu, xtanh, sigmoid
    + u: uncentered data. Can be f for False, or otherwise for True
    + n: noisy data. Can be f for False, or otherwise for True
* -ft Flow type used for the iFlow model. Can be set to RQNSF_AG, RQNSF_C or PlanarFlow (default: RQNSF_AG)
* -nph Determines how the natural parameters (λ(u)) is implemented in iFlow. Can be set to original (paper's implementation), removed (removing λ(u), resulting in Flow), or fixed (allowing for more expressivity by only applying the activation layer on the η's) (default: original)
* -s Seed used for training the model (default: 1)
* -c Trains network on GPU if set
* -p Preloads data on GPU if set
* -sr: Saves the mean correlation coefficient (MCC) scores to a JSON file, where each key is a string representing the netwerk configuration and the value is a list with MCC scores with the seed as index. Can be set to 'data' or 'model', depending on which seed is varied. (default: None)

In all our experiments -x was set to 1000_40_5_5_3_1_gauss_xtanh_u_f and the arguments -c and -p were set.

To train the iFlow model as intended by the original paper on seed=1 run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p
```

And for iVAE:
```train
python main.py -i iVAE -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p
```

To train regular Normalized Flow (so without the natural parameters) run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p -nph removed
```

To train iFlow using (the less complex) PlanarFlow run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p -ft PlanarFlow
```

To train regular Normalized Flow (so without the natural parameters) using PlanarFlow run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p -ft PlanarFlow -nph removed
```

To train iFlow using the more flexible ("fixed") λ(u) implementation run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p -nph fixed
```

To train the models for different seeds, see and run the bash files iFlow.sh and iVAE.sh for different configurations of varying the seed, aswell as saving the results in a JSON file based on seed index.


### Half-Moon Dataset

To get more insight into how the iFlow model differs from the standard Flow model the files ```real-nvp-pytorch-iflow.ipynb``` and ```real-nvp-pytorch-flow.ipynb``` contain code to train an iFlow or regular Flow model using the RealNVP normalized flow network on the half-moon dataset. Run these Jupyter Notebook files to train and save the models.

## Pre-trained Models

Two pre-trained models for the Half-moon dataset are located in ```trained_models/``` 


## Result evaluation

The Jupyter Notebook file ```visualize_results.ipynb``` contains code to visualize the results from both the TCL dataset aswell as the Half-Moon dataset experiments. This Jupyter Notebook file can simply be executed from top-to-bottom. Note that for the Notebook to work as intended the provided results in ```results/results_variable_dataseed.json```and ```results/results_variable_netseed.json``` and the pre-trained Half-Moon networks should be available or, alternatively, these results and models are re-generated using the Half-Moon dataset training notebooks and/or executing the different configurations in the ```iFlow.sh``` and ```iVAE.sh``` files.


## Results

The following results were obtained by running the visualization notebook for the TCL dataset using the provided results, which in turn were generated using the .sh scripts.


| Model name         | Average MCC  | Standard deviation |
| ------------------ |---------------- | -------------- |
| iFlow   |     0.7131         |      0.0585       |
| iVAE (SplineFlow)   |     0.4701         |      0.0726       |
| Flow (SplineFlow)  |     0.6446         |      0.0638      |
| iFlow (PlanarFlow)  |     0.619        |      0.0421       |
| Flow (PlanarFlow)  |     0.5786      |      0.0522       |
| "improved" iFlow (SplineFlow)  |     0.7499        |      0.0804       |

