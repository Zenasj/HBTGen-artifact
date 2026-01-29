# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_cells, in_features, out_features, activation_class):
        super(MyModel, self).__init__()
        layers = []
        last_dim = in_features
        for i in range(len(num_cells)):
            layers.append(nn.Linear(last_dim, num_cells[i]))
            layers.append(activation_class())
            last_dim = num_cells[i]
        layers.append(nn.Linear(last_dim, out_features))
        self.mlp = nn.Sequential(*layers)
        self.critic_1 = self.mlp
        self.critic_2 = self.mlp

    def forward(self, x):
        y_separate = torch.stack([self.critic_1(x), self.critic_2(x)])
        critic_params = {k: torch.stack([self.critic_1.state_dict()[k], self.critic_2.state_dict()[k]]) for k in self.critic_1.state_dict()}
        critic_call = lambda params, inputs: torch.func.functional_call(self.critic_1, params, inputs)
        y_vmap = torch.vmap(critic_call, (0, None), randomness="same")(critic_params, x)
        return y_separate, y_vmap

def my_model_function():
    mlp_kwargs = {
        "num_cells": [256, 256, 256],
        "in_features": 30,
        "out_features": 1,
        "activation_class": nn.ReLU,
    }
    return MyModel(**mlp_kwargs)

def GetInput():
    batch_size = 4096
    in_features = 30
    return torch.randn(batch_size, in_features)

