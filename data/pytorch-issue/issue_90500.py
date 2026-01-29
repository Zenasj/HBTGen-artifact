# torch.rand(3, 5, 4, dtype=torch.float32)  # Inferred input shape (B, T, C)

import torch
from torch.nn.utils.stateless import functional_call
import torch.autograd as autograd
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, action_dim, z_dim, skill_length):
        super().__init__()
        self.lin1 = nn.Linear(action_dim, action_dim)
        self.lstm = nn.LSTM(input_size=action_dim, hidden_size=z_dim, batch_first=True)
        self.lin2 = nn.Linear(z_dim, z_dim)

    def forward(self, skill):
        a, b, c = skill.shape
        skill = skill.reshape(-1, skill.shape[-1])
        embed = self.lin1(skill)
        embed = embed.reshape(a, b, c)
        mean, _ = self.lstm(embed)
        mean = mean[:, -1, :]
        mean = self.lin2(mean)
        return mean

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel(4, 2, 5)
    params = {}
    for name, param in model.named_parameters():
        if len(param.shape) == 1:
            init = torch.nn.init.constant_(param, 0)
        else:
            init = torch.nn.init.orthogonal_(param)
        params[name] = nn.Parameter(init)
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(3, 5, 4, dtype=torch.float32)

