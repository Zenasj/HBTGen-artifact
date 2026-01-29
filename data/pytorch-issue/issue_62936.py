# torch.rand(16, 1, 8, dtype=torch.float32)  # Inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, dim=8, batch_axis=1, sequence_length=16):
        super().__init__()
        self.dim = dim
        self.batch_axis = batch_axis
        self.sequence_length = sequence_length
        self.temperature = 10000

    @torch.no_grad()
    def gen_pos_embedding(self, sl, dim, device):
        pe = torch.zeros(sl, dim, device=device)
        position = torch.arange(0, sl, device=device).unsqueeze(1)
        div_term = self.temperature**-(torch.arange(0, dim, 2, device=device) / dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x is dim of [Sl, B, D]
        sl = x.size(0) if self.batch_axis in (-1, 1) else x.size(1)
        embedding = self.gen_pos_embedding(sl, self.dim, x.device)

        if self.batch_axis > -1:
            embedding = embedding.unsqueeze(self.batch_axis)
            # Use repeat_interleave for dynamic input case
            embedding = embedding.repeat_interleave(x.size(self.batch_axis), self.batch_axis)

        return x + embedding, embedding

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(16, 1, 8, dtype=torch.float32)

