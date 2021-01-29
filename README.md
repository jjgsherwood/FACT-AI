<!-- >📋  A template README.md for code accompanying a Machine Learning paper -->
# [Re] Identifying Through Flows for RecoveringLatent Representations

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

In our research numerous configurations of the model has been trained and evaluated. These models were trained across two different datasets: the original TLC dataset and the Half-Moon dataset, as provided by Sklearn.

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
* -sr Saves the mean correlation coefficient (MCC) scores to a JSON file, where each key is a string representing the netwerk configuration and the value is a list with MCC scores with the seed as index. Can be set to 'data' or 'model', depending on which seed is varied. (default: None)
* -sm Saves the model in trained_models/ if provided.

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

To train the models for different seeds, see and run the bash files `iFlow.sh` and `iVAE.sh` for different configurations of varying the seed, aswell as saving the results in a JSON file based on seed index. 
For Windows, either run these .sh files in a suitable environment (such as the Git bash terminal), or adapt the `run_windows_iflow.bat` file to your need and execute.

To save the models in `trained_models/`, add the ```-sm``` argument.

All these model configurations were trained using on an Nvidia RTX 2080Ti with 11GB of VRAM (4GB should be plenty). The table below shows the different run-times for each configuration.

| Model name         | Totall training time (20 epochs) |
| ------------------ |---------------- |
| iFlow (SplineFlow)  |     20 minutes        |
| iVAE    |     75 seconds         |
| Flow (SplineFlow)  |     19 minutes         |
| iFlow (PlanarFlow)  |     7 minutes        |
| Flow (PlanarFlow)  |     5 minutes      |
| "improved" iFlow (SplineFlow)  |     20 minutes        |



### Half-Moon Dataset

To get more insight into how the iFlow model differs from the standard Flow model the files ```real-nvp-pytorch-iflow.ipynb``` and ```real-nvp-pytorch-flow.ipynb``` contain code to train an iFlow or regular Flow model using the RealNVP normalized flow network on the half-moon dataset. Run these Jupyter Notebook files to train and save the models.

In our research, these models were trained on an AMD Ryzen 9 3900X 12 core/24 thread CPU with 32GB DDR4 3200MHz RAM. Each model took around 3 minutes to train.

## Pre-trained Models

Two pre-trained models for the Half-moon dataset are located in ```trained_models/``` 

## Gaussian Blob dataset
In addition to our research we also include our own created Gaussian Blob dataset in `blob/Gaussian_blob.ipynb`. We have not performed any tests regarding the networks on this dataset, but the dataset is provided for possible use.


## Result evaluation

The Jupyter Notebook file ```visualize_results.ipynb``` contains code to visualize the results from both the TCL dataset aswell as the Half-Moon dataset experiments. This Jupyter Notebook file can simply be executed from top-to-bottom, without needing a GPU. Note that for the Notebook to work as intended the provided results in ```results/results_variable_dataseed.json```and ```results/results_variable_netseed.json``` and the pre-trained Half-Moon networks should be available or, alternatively, these results and models are re-generated using the Half-Moon dataset training notebooks and/or executing the different configurations in the ```iFlow.sh``` and ```iVAE.sh``` files.


## Results

### TCL Dataset
The following results were obtained by running the visualization notebook for the TCL dataset using the provided results, which in turn were generated using the .sh scripts.

| Model name         | Average MCC  | Standard deviation |
| ------------------ |---------------- | -------------- |
| iFlow (SplineFlow)   |     0.7131         |      0.0585       |
| iVAE  |     0.4701         |      0.0726       |
| Flow (SplineFlow)  |     0.6446         |      0.0638      |
| iFlow (PlanarFlow)  |     0.619        |      0.0421       |
| Flow (PlanarFlow)  |     0.5786      |      0.0522       |
| "improved" iFlow (SplineFlow)  |     0.7499        |      0.0804       |


### Half-Moon Dataset
The following results were achieved with our iFlow vs Flow visualization experiments.


![Model spaces](https://github.com/jjgsherwood/FACT-AI/blob/main/results/flow_en_de_report.png)

This image shows the original data space X, the latent space generated by encoding the data using the model, the generated latent space according to approximate priors and the decoded generated latent space in the original data space for both iFlow and Flow.

![Model spaces](https://github.com/jjgsherwood/FACT-AI/blob/main/results/flow_latent_report_mini.png)

And this image shows latent space exploration by decoding the entire latent space into data space for both iFlow and Flow.

