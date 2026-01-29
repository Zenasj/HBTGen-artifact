# torch.rand(B, C) ← Add a comment line at the top with the inferred input shape
import torch
from torch import nn

in_dim = 512
layer_dims = [512, 1024, 256]
out_dim = 10

# Single layer definition
class MyNetworkBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


# Full model definition
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = len(layer_dims)
        prev_dim = in_dim
        # Add layers one by one
        for i, dim in enumerate(layer_dims):
            self.add_module(f"layer{i}", MyNetworkBlock(prev_dim, dim))
            prev_dim = dim

        # Final output layer (with OUT_DIM projection classes)
        self.output_proj = nn.Linear(layer_dims[-1], out_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i}")
            x = layer(x)

        return self.output_proj(x)


def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 32
    input_tensor = torch.randn(batch_size, in_dim)
    return input_tensor

