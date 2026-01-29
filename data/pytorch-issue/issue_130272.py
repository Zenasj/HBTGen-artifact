# Input: (values (shape: (N, 64), dtype=torch.bfloat16, requires_grad=True), offsets (shape: (17,), dtype=torch.int32))
import torch
from torch import nn
from torch.nested._internal.nested_tensor import nested_view_from_values_offsets

def get_values_offsets(batch_size=16, max_seqlen=200, inner_dim=64):
    lengths = torch.randint(1, max_seqlen, (batch_size,), dtype=torch.int32)
    offsets = torch.zeros((batch_size + 1,), dtype=torch.int32)
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    values = torch.randn((offsets[-1].item(), inner_dim), dtype=torch.bfloat16, requires_grad=True)
    return values, offsets

class MyModel(nn.Module):
    def forward(self, inputs):
        values, offsets = inputs
        return nested_view_from_values_offsets(values, offsets, max_seqlen=200)

def my_model_function():
    return MyModel()

def GetInput():
    return get_values_offsets()

