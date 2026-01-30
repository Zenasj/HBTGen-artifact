import torch.nn as nn

py
import math

import torch
from torch import nn, Tensor


class PoseFSQAutoEncoder(nn.Module):
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


model = PoseFSQAutoEncoder()
batch = torch.randn(size=(2, 10, 178, 3))

model.eval()  # <--- HERE: Different behavior .train() vs .eval()

with torch.no_grad():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(batch)