# 线速度估计器
from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.parametrizations import spectral_norm

# 简单的前馈神经网络,传入历史数据估计优先观测
class MYEstimator(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu",
                        **kwargs):
        super(MYEstimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        activation = get_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)
        print(f"Estimator MLP: {self.estimator}")
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)