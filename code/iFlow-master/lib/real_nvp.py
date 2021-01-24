import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams

import torch
from torch import nn
from torch.nn.parameter import Parameter

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler


class RealNVP(nn.Module):
    def __init__(self, x_dim, hidden_dim, flow_length, prior):
        super(RealNVP, self).__init__()
        self.prior = prior
        mask = np.array([[1 if (x_dim-j-1)==i else 0 for i in range(x_dim)] for j in range(x_dim)] * (flow_length//x_dim))
        self.mask = nn.Parameter(torch.from_numpy(mask.astype(np.float32)), requires_grad=False)
        nett = lambda: nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, x_dim))
        nets = lambda: nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, x_dim), nn.Sigmoid())
        self.t = torch.nn.ModuleList([nett() for _ in range(flow_length)])
        self.s = torch.nn.ModuleList([nets() for _ in range(flow_length)])

    def forward(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J