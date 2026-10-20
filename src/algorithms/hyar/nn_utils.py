"""Helpers from upstream HyAR (embedding/Utils/utils.py) used by the action representation.

Copied verbatim from TJU-DRL-LAB/self-supervised-rl; see LICENSE and NOTICE.md in this folder.
"""
import numpy as np
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.ctr = 0
        self.nan_check_fequency = 10000

    def custom_weight_init(self):
        # Initialize the weight values
        for m in self.modules():
            weight_init(m)

    def update(self, loss, retain_graph=False, clip_norm=False):
        self.optim.zero_grad()  # Reset the gradients
        loss.backward(retain_graph=retain_graph)
        self.step(clip_norm)

    def step(self, clip_norm):
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        self.optim.step()
        self.check_nan()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for name, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(name + ": Weights have become nan... Exiting.")

    def reset(self):
        return



def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
    Computationally more expensive? Maybe, Not sure.
    adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    # print("x",x)
    # print("y",y)

    x_norm = (x ** 2).sum(1).view(-1, 1)   #sum(1)将一个矩阵的每一行向量相加
    y_norm = (y ** 2).sum(1).view(1, -1)
    # print("x_norm",x_norm)
    # print("y_norm",y_norm)
    y_t = torch.transpose(y, 0, 1)  #交换一个tensor的两个维度
    # a^2 + b^2 - 2ab
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)    #torch.mm 矩阵a和b矩阵相乘
    # dist[dist != dist] = 0 # replace nan values with 0
    # print("dist",dist)
    return dist


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = 0#.1/ np.sqrt((fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        # m.weight.data.normal_(0.0, 0.03)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
