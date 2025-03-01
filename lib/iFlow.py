from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch import distributions

import pdb

#THIS IS SUPPOSEDLY TO BE A WRONG IMPLEMENTATION SINCE SOFTPLUS IS EXERTED ON ETA!!!

################# iFlow ####################
import lib
from lib.rq_spline_flow import utils as utils_rqsf
from lib.rq_spline_flow import transforms
from lib.rq_spline_flow import nn_

def create_base_transform(i, base_transform_type, dim, num_bins):
    if base_transform_type == 'affine':
        return transforms.AffineCouplingTransform(
            mask=utils_rqsf.create_alternating_binary_mask(
                        features=dim,
                        even=(i % 2 == 0)),
                        transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                                                                           in_features=in_features,
                                                                           out_features=out_features,
                                                                           hidden_features=32,
                                                                           num_blocks=2,
                                                                           use_batch_norm=True)
        )

    elif base_transform_type == "rqsf_c":
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils_rqsf.create_alternating_binary_mask(
                        features=dim,
                        even=(i % 2 == 0)),
                        transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                                                                           in_features=in_features,
                                                                           out_features=out_features,
                                                                           hidden_features=32,
                                                                           num_blocks=2,
                                                                           use_batch_norm=True),
                        tails='linear',
                        tail_bound=5,
                        num_bins=num_bins,
                        apply_unconditional_transform=False
        )

    elif base_transform_type == "rqsf_ag":
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=dim,
            hidden_features=256,
            context_features=None,
            num_bins=num_bins,
            tails='linear',
            tail_bound=3,
            num_blocks=2,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=0.25,
            use_batch_norm=0
        )

    else:
        raise ValueError


def create_linear_transform(linear_transform_type, dim):
    if linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=dim)
    elif linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=dim),
            transforms.LULinear(dim, identity_init=True)
        ])
    elif linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=dim),
            transforms.SVDLinear(dim, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError


def create_transform(dim, num_flow_steps, num_bins):
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform("lu", dim),
            create_base_transform(i, "rqsf_ag", dim, num_bins)
        ]) for i in range(num_flow_steps)
    ] + [
        create_linear_transform("lu", dim)
    ])
    return transform


from lib.planar_flow import *
from lib.rq_spline_flow import *
from lib.rq_spline_flow.rq_spline_flow import *

class iFlow(nn.Module):
    def __init__(self, args):
        super(iFlow, self).__init__()

        self.args = args
        self.bs = args['batch_size']
        assert args['latent_dim'] == args['data_dim']
        self.x_dim = self.z_dim = args['latent_dim']
        self.u_dim = args['aux_dim']
        self.nat_param_method = args['nat_param_method']
        self.k = 2 # number of orders of sufficient statistics

        flow_type = args['flow_type']

        if flow_type == "PlanarFlow":
            self.nf = NormalizingFlow(dim=self.x_dim, flow_length=args['flow_length'])

        elif flow_type == "RQNSF_C":
            transform = transforms.CompositeTransform([
                create_base_transform(i, "rqsf_c", self.z_dim, 64) for i in range(args['flow_length'])
            ])
            self.nf = SplineFlow(transform)

        elif flow_type == "RQNSF_AG":
            transform = create_transform(self.z_dim, args['flow_length'], args['num_bins'])
            self.nf = SplineFlow(transform)

        else:
            raise ValueError

        self.feb = FreeEnergyBound(args=args)

        str2act = {"Sigmoid": nn.Sigmoid(),
                   "ReLU": nn.ReLU(inplace=True),
                   "Softmax": nn.Softmax(),
                   "Softplus": nn.Softplus()}

        self.max_act_val = None
        act_str = args['nat_param_act']
        if act_str in str2act:
            nat_param_act = str2act[args['nat_param_act']]
        else:
            assert act_str.startswith("Sigmoidx")
            nat_param_act = nn.Sigmoid()
            self.max_act_val = float(act_str.split("x")[-1])
        self.nat_param_act = nat_param_act

        if self.nat_param_method != "removed":
            if self.u_dim == 40:
                self._lambda = nn.Sequential(
                    nn.Linear(self.u_dim, 30),
                    nn.ReLU(inplace=True),
                    nn.Linear(30, 20),
                    nn.ReLU(inplace=True),
                    nn.Linear(20, 2*self.z_dim),
                ) ## for self.u_dim == 40
            elif self.u_dim == 3:
                self._lambda = nn.Sequential(
                    nn.Linear(self.u_dim, 6),
                    nn.ReLU(inplace=True),
                    nn.Linear(6, 5),
                    nn.ReLU(inplace=True),
                    nn.Linear(5, 2*self.z_dim),
                ) ## for self.u_dim == 60
            elif self.u_dim == 60:
                self._lambda = nn.Sequential(
                    nn.Linear(self.u_dim, 45),
                    nn.ReLU(inplace=True),
                    nn.Linear(45, 25),
                    nn.ReLU(inplace=True),
                    nn.Linear(25, 2*self.z_dim),
                ) ## for self.u_dim == 60

        #assert self.u_dim == 5
        #self._lambda = nn.Sequential(
        #    nn.Linear(self.u_dim, 4),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(4, 4),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(4, 2*self.z_dim),
        #    nat_param_act,
        #) ## for visualisation where self.u_dim == 5

        self.set_mask(self.bs)

    def set_mask(self, bs=64):
        #col1 = torch.ones((bs, self.z_dim, 1)) * 1e-5
        #col2 = torch.zeros((bs, self.z_dim, 1))
        #self.mask1 = torch.cat((col1, col2), axis=2).to(self.args['device'])
        self.mask2 = torch.ones((bs, self.z_dim, 2)).to(self.args['device'])
        self.mask2[:, :, 0] *= -1.0

    def forward(self, x, u):
        B = x.size(0)
        z, log_jacobians = self.nf(x)
        # z is of shape [64, 2]
        # log_jacobians is a list of length 16, each of which is of shape [64, 1]

        # construct T
        # of shape (B, k, n), where n is latent dim and k is parameter of exponential family.
        T = torch.cat((z*z, z), axis=1).view(B, self.k, -1)

        # construct \lambda based on implementation method.
        # of shape (B, n, k).
        if self.nat_param_method != "removed":
            nat_params = self._lambda(u)
            nat_params = nat_params.reshape(B, self.z_dim, 2) #+ 1e-5 # force the natural_params to be strictly > 0.
            if self.nat_param_method == "original":
                nat_params = self.nat_param_act(nat_params)
            # Only applies softplus over the xis and not etas to allow for more flexibility.   
            elif self.nat_param_method == "fixed":
                xi = self.nat_param_act(nat_params[:,:,0].unsqueeze(2))
                nat_params = torch.cat((xi, nat_params[:,:,1].unsqueeze(2)), 2)
            else:
                raise ValueError
        # Effectively disables the contribution and training of the Lambda network to test non-i Flow models. 
        else:
            nat_params = torch.ones((B, self.z_dim, 2)).to(self.args['device'])
        if self.max_act_val:
            nat_params = nat_params * self.max_act_val #+ 1e-5 #self.mask1
        nat_params = nat_params * self.mask2
        return z, T, nat_params, log_jacobians

    def neg_log_likelihood(self, x, u):
        z_est, T, nat_params, log_jacobians = self.forward(x, u)
        return self.feb(T, nat_params, log_jacobians), z_est

    def inference(self, x, u):
        z_est, _, nat_params, _ = self.forward(x, u)
        return z_est, nat_params


class FreeEnergyBound(nn.Module):
    def __init__(self, args):
        super(FreeEnergyBound, self).__init__()
        #self.bs = args['batch_size']
        self.x_dim = self.z_dim = args['latent_dim']
        self.u_dim = args['aux_dim']

    def forward(self, T, nat_params, log_jacobians):
        B = T.size(0)
        sum_of_log_jacobians = torch.sum(log_jacobians)

        sum_traces = 0.0
        for i in range(B):
            sum_traces += (torch.trace(nat_params[i].mm(T[i]))) # no .t(), since it is nxk-by-kxn matrix multiplication
        avg_traces = sum_traces / B

        log_normalizer = -.5 * torch.sum(torch.log(torch.abs(nat_params[:, :, 0]))) / B
        nat_params_sqr = torch.pow(nat_params[:, :, 1], 2) # of shape [B, n]
        log_normalizer -= (torch.sum(nat_params_sqr / (4*nat_params[:, :, 0])) / B)

        #loss = log_normalizer + (-sum_of_log_jacobians).mean() - avg_traces

        neg_trace = avg_traces.mul(-1)
        neg_log_det = torch.sum(sum_of_log_jacobians.mul(-1)) / B

        return log_normalizer, neg_trace, neg_log_det
