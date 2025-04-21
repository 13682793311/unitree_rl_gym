# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch 
import torch.nn as nn  # torch的神经网络模块nn，用于构建深度神经网络
from torch.distributions import Normal
from torch.nn.modules import rnn

# 加入历史观测编码器
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps     # 历史时间步的数量

        channel_size = 10
        # last_activation = nn.ELU()

        # 全连接层
        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )   # 输入为状态信息，输出为3 * channel_size维的向量

        # 根据时间步的不同，选择不同的卷积层结构
        # 使用一维卷积层处理时间序列数据
        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))
        # 最终输出层
        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    # 前向传播
    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

# 将历史观测编码器改为简单的前馈神经网络,使用历史编码器去估计优先观测和地形信息
class StateHistoryMLP(nn.Module):
    def __init__(self,
                 activation,  
                 input_size,
                 tsteps,
                 output_size,
                 hidden_dims=[256, 128],
                 tanh_encoder_output=False       
                 ):
        super(StateHistoryMLP, self).__init__()

        self.input_dim = input_size * tsteps
        self.output_dim = output_size
        self.tsteps = tsteps
        self.activation = activation
        HistoryEncoder_layers = []
        HistoryEncoder_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        HistoryEncoder_layers.append(self.activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                HistoryEncoder_layers.append(nn.Linear(hidden_dims[l], self.output_dim))
            else:
                HistoryEncoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                HistoryEncoder_layers.append(self.activation)
        # estimator_layers.append(nn.Tanh())
        self.HistoryEncoder = nn.Sequential(* HistoryEncoder_layers)
        #print(f" HistoryEncoder: {self.HistoryEncoder}")
    
    # 前向传播函数
    def forward(self, obs):
        obs = obs.view(obs.shape[0], -1)  # 将输入展平为一维
        output = self.HistoryEncoder(obs)  # 通过全连接层进行前向传播
        return output  # 返回输出结果



# 改写actor类
class Actor(nn.Module):
    def __init__(self, num_prop,     
                 num_scan,
                 num_actions, 
                 actor_hidden_dims,   # actor网络隐藏层的维度
                 priv_encoder_dims,   # 特权状态编码器的维度
                 scan_encoder_dims,   # 地形特征编码器的维度
                 num_priv_latent,     # 潜在特征
                 num_hist, activation, # 历史观测的维度
                 tanh_encoder_output=False) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent   # 潜在特征
        self.if_scan_encode =  scan_encoder_dims is not None and num_scan >0
        #self.if_scan_encode =  scan_encoder_dims is not None
        self.num_scan = num_scan
        # 构建特权状态编码器
        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        # 构建地形特征编码器
        if len(scan_encoder_dims)>0:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            scan_encoder_output_dim = num_scan
            

        #self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)
        self.history_encoder = StateHistoryMLP(activation, num_prop, num_hist, priv_encoder_output_dim + scan_encoder_output_dim)

        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+
                                      priv_encoder_output_dim+
                                      scan_encoder_output_dim, 
                                      actor_hidden_dims[0]))  # 普通观测 + 优先观测编码向量 + 扫描点编码向量
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    # 动作生成
    def forward(self, obs, hist_encoding: bool, scandots_latent=None):
        obs_prop = obs[:, :self.num_prop]
        # 如果使用历史编码器，则直接输入
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
            # 将普通观测和历史预测拼接作为输入
            backbone_input = torch.cat([obs_prop, latent], dim=1)
        # 否则将使用特权观测和地形信息
        else: 
            # 是否使用扫描点编码器
            if self.if_scan_encode:
                if scandots_latent is None:
                    scan_latent = self.infer_scan_latent(obs)
                else:
                    scan_latent = scandots_latent
                priv_latent = self.infer_priv_latent(obs)
                backbone_input = torch.cat([obs_prop, priv_latent, scan_latent], dim=1) 
            else:
                scan_latent = torch.zeros(obs.shape[0], self.num_scan, device=obs.device)
                priv_latent = self.infer_priv_latent(obs)
                scan_latent = torch.zeros(obs.shape[0], self.num_scan, device=obs.device)  # 确保维度匹配
                backbone_input = torch.cat([obs_prop, priv_latent, scan_latent], dim=1)  
        backbone_output = self.actor_backbone(backbone_input)        
        return backbone_output
        
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop : self.num_prop + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scan_latent(self,obs):
        scan = obs[:, self.num_prop + self.num_priv_latent : self.num_prop + self.num_priv_latent + self.num_scan]
        return self.scan_encoder(scan)

    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist)
    

# 定义ActorCritic网络结构
class MYActorCritic(nn.Module):
    # 当前模型是否采用循环结构
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        scan_encoder_dims= [256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MYActorCritic, self).__init__()

        # 激活函数获取
        activation = get_activation(activation)

        priv_encoder_dims= kwargs['priv_encoder_dims']

        # actor网络的构建
        self.actor = Actor(num_prop, num_scan, num_actions, actor_hidden_dims, priv_encoder_dims,scan_encoder_dims, num_priv_latent, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])

        # Value function
        # 定义critic网络结构
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        # 输出动作噪声
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # 更新概率分布
    def update_distribution(self, observations,hist_encoding):
        mean = self.actor(observations,hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    # 更新动作分布后，从正态分布中采样一个动作返回
    def act(self, observations,hist_encoding=False, **kwargs):
        self.update_distribution(observations,hist_encoding)
        return self.distribution.sample()
    
    # 计算给定动作在当前分布下的对数概率
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # 在推理时直接返回actor网络的输出均值
    def act_inference(self, observations,hist_encoding=False, scandots_latent=None):
        if not eval:
            actions_mean = self.actor(observations,hist_encoding,scandots_latent)
            return actions_mean
        else:
            actions_mean, latent_history, latent_priv = self.actor(observations,hist_encoding,scandots_latent)
            return actions_mean


    # 通过critic网络对输入进行评估，返回状态值
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
