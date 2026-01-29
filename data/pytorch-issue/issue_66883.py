# torch.rand(32, 256, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense_arch = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(256)
        self.linear_split_1 = nn.Linear(256, 256)
        self.linear = nn.Linear(256, 256)

    def forward(self, x):
        dense_out = self.dense_arch(x)
        norm = self.layer_norm(dense_out)
        split_sum = self.linear_split_1(norm)
        prod = self.linear(split_sum) * norm
        interaction = torch.sum(torch.stack([norm, prod], dim=0), dim=0)
        return interaction

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(32, 256, dtype=torch.float32)

