# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.submod = torch.jit.script(testSubMod(rnn_dims))

    def forward(self, x):
        out = torch.ones(
            [
                x.size(0),
                x.size(1)
            ],
            dtype=x.dtype,
            device=x.device
        )
        return self.submod(out)

class testSubMod(nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.lin = torch.nn.Linear(32, 32, bias=True)  # This bit here!

    def forward(self, out):
        for _ in torch.arange(8):
            out = self.lin(out)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    input_data = torch.ones((32, 32, 32)).float()
    return input_data

