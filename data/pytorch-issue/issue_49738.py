# torch.rand(1, 32, 32, 32, dtype=torch.float32)  # Inferred input shape (B, H, W, C)

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, 1)
        self.norm = nn.LayerNorm(out_size)  # Adjusted to match the output size of the conv layer

    def forward(self, inp):
        # Checkpoint the convolutional layer
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        output = checkpoint(create_custom_forward(self.conv), inp)
        output = output.permute(0, 2, 3, 1)  # permute so we have channel in the end
        output = checkpoint(create_custom_forward(self.norm), output)
        return output

def my_model_function():
    in_size = 32
    out_size = 64
    return MyModel(in_size, out_size)

def GetInput():
    in_size = 32
    return torch.randn([1, 32, 32, in_size], requires_grad=True).cuda()

