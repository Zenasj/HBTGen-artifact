# torch.rand(B, 2, 4, 10, dtype=torch.float32)  # Example input shape (batch, channels, seq_len, head_dim)
import torch
from torch import nn
from collections import namedtuple
import functools

SdpaShape = namedtuple('Sdpa_Shape', ['batch', 'num_heads', 'seq_len', 'head_dim'])

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = SdpaShape(2, 4, 10, 32)  # Example shape parameters
        # Use functools.partial to replicate Dynamo's problematic scenario
        self.xlogy_func = functools.partial(torch.xlogy, other=torch.tensor(2.0))

    def forward(self, x):
        # Reshape input to match SdpaShape dimensions
        x_reshaped = x.view(
            self.shape.batch,
            self.shape.num_heads,
            self.shape.seq_len,
            self.shape.head_dim
        )
        return self.xlogy_func(x_reshaped)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the model's expected dimensions
    batch, num_heads, seq_len, head_dim = 2, 4, 10, 32
    input_shape = (batch * num_heads * seq_len * head_dim,)
    return torch.rand(input_shape, dtype=torch.float32)  # Common dtype for Dynamo testing

