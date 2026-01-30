import random

import torch

class Model(torch.nn.Module):
    def forward(self, inputs):
        x = inputs[0].detach().requires_grad_()
        with torch.enable_grad():
            loss = torch.sqrt(x).sum()
            return torch.autograd.grad([loss], [x])[0]

torch.jit.trace(Model(), torch.FloatTensor(2, 3, 4).uniform_(0, 10))

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEFAULT_DEVICE = torch.device('cpu')

def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)
    
def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

# had to define this because torch.diag_embed was not being onnx exported
def custom_diag_embed(diagonal):
    batch_size, dim = diagonal.shape
    return torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1).to(diagonal.device) * diagonal.unsqueeze(-1)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, state_min = None, state_max = None, state_mean = None, state_std = None):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim], output_activation=nn.Tanh)
        self.state_min = state_min
        self.state_max = state_max
        self.state_mean = state_mean
        self.state_std = state_std

    def forward(self, obs):
        if self.state_mean is not None:
            obs = (obs - self.state_mean) / self.state_std
        mean, log_std = torch.chunk(self.net(obs), 2, dim=-1)
        std = torch.exp(log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        # scale_tril = torch.diag(std)
        # scale_tril = torch.diag_embed(std)
        scale_tril = custom_diag_embed(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=True, enable_grad=False): # only called during evaluation
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs) # dist is a MultivariateNormal object, calls forward
            return dist.mean, dist.stddev if deterministic else dist.sample()

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2, state_min = None, state_max = None, state_mean = None, state_std = None):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, activation=nn.Tanh, squeeze_output=True)
        self.q2 = mlp(dims, activation=nn.Tanh, squeeze_output=True)
        self.state_min = state_min
        self.state_max = state_max
        self.state_mean = state_mean
        self.state_std = state_std

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        # sa = sa.float() # added this line
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            if self.state_mean is not None:
                state = (state - self.state_mean) / self.state_std
            return torch.min(*self.both(state, action))

class QFGrad(nn.Module):
    def __init__(self, policy, qf, beta=0):
        super().__init__()
        self.policy = policy
        self.qf = qf
        self.beta = beta

    def forward(self, state, hidden_state, cell_state):
        state = state.squeeze(1)
        action, _ = self.policy.act(state, enable_grad=True)        
        updated_action = action + self.beta * self.calc_q_grad(state, action)
        return updated_action, hidden_state, cell_state

    def calc_q_grad(self, state, action):
        qf_output = self.qf(state, action, enable_grad=True)
        # return torch.autograd.functional.jacobian(self.qf, (state, action), create_graph=False)[1]
        return torch.autograd.grad(qf_output, action, create_graph=True, retain_graph=True)[0]

def main():

    obs_dim = 150
    act_dim = 1

    policy = GaussianPolicy(obs_dim, act_dim * 2, hidden_dim=128, n_hidden=2)
    qf = TwinQ(obs_dim, act_dim, hidden_dim=128, n_hidden=2)

    policy.to(DEFAULT_DEVICE)
    qf.to(DEFAULT_DEVICE)

    # create a meta model with policy and qf, that follows the equation: output = policy(input) + grad_a * qf(input, policy(input))
    meta_model = QFGrad(policy, qf, beta=0.05)

    # Example usage
    # input_tensor = torch.from_numpy(np.random.rand(1, 1, 150)).float()
    input_tensor = torchify(np.random.rand(1, 1, 150))
    dummy_hidden = torch.zeros(1, 1)
    dummy_cell = torch.zeros(1, 1)
    
    # Export to ONNX with additional inputs and outputs
    torch.onnx.export(meta_model, 
                  (input_tensor, dummy_hidden, dummy_cell),
                  "demo_1.onnx", 
                  input_names=["obs", "hidden_states", "cell_states"],
                  output_names=["output", "state_out", "cell_out"],)
    
if __name__ == '__main__':
    main()