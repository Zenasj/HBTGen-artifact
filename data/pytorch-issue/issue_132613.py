# torch.rand(2, 10, 178, 3, dtype=torch.float32)  # Input shape and dtype

import math
import torch
from torch import nn, Tensor

class MyModel(nn.Module):
    def __init__(self, pose_dims: tuple = (178, 3), hidden_dim=512, nhead=16, dim_feedforward=2048, num_layers=6):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(math.prod(pose_dims), hidden_dim, bias=False),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
                ),
                num_layers=num_layers,
            ),
        )

    def forward(self, batch: Tensor):
        return self.encoder(batch)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(size=(2, 10, 178, 3), dtype=torch.float32)

