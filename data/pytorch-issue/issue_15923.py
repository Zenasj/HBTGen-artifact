# torch.rand(6, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, jac_t, state, dt):
        super().__init__()
        self.register_buffer('jac_t', jac_t)  # Fixed parameter (6, n_actions)
        self.register_buffer('state', state)   # Fixed parameter (6, 1)
        self.dt = dt
        self.q_dot = nn.Parameter(torch.randn(jac_t.shape[1], 1))  # Learnable parameter (n_actions, 1)

    def forward(self, target):
        next_state = torch.matmul(self.jac_t, self.q_dot) * self.dt + self.state
        loss = torch.pow(next_state - target, 2).sum()
        return loss

def my_model_function():
    n_actions = 22  # From Python example
    dt = 0.01
    jac_t = torch.randn(6, n_actions)  # Matches Python example's jac_t shape
    state = torch.randn(6, 1)
    return MyModel(jac_t, state, dt)

def GetInput():
    # Returns a random tensor matching target shape (6,1) from Python example
    return torch.randn(6, 1, dtype=torch.float32)

