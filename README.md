<!-- >📋  A template README.md for code accompanying a Machine Learning paper -->
# My Paper Title

This Github repository is part of our submission to the 2021 MLRC Reproducibility Challenge, where we attempt to reproduce the results and verify the claims of [Identifying through Flows for Recovering Latent Representations](https://arxiv.org/abs/1909.12555). This repository is a modified version of the official [GitHub implementation](https://github.com/MathsXDC/iFlow).

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements for training on the GPU:

```setup
pip install -r requirements_GPU.txt
```

And for training on the CPU:

```setup
pip install -r requirements_CPU.txt
```

## Training

In our research numerous configurations of the model has been trained. To train the different models, there are several important parameters:

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

To train Flow (so without the natural parameters) run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p -nph removed
```

To train iFlow using (the less complex) PlanarFlow run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p -ft PlanarFlow
```

To train Flow (so without the natural parameters) using PlanarFlow run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p -ft PlanarFlow -nph removed
```

To train iFlow using the more flexible ("fixed") λ(u) implementation run:
```train
python main.py -x 1000_40_5_5_3_1_gauss_xtanh_u_f -c -p -nph fixed
```

To train the models for different seeds, see and run the bash files iFlow.sh and iVAE.sh for different configurations of varying the seed.

## Result evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.
