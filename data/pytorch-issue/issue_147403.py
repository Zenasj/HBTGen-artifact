# torch.rand(5), torch.tensor([0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, lengths = inputs
        size = list(lengths.size())
        size[0] //= 2  # Modify size as in example
        stride = list(lengths.stride())
        stride[0] *= 2  # Modify stride as in example
        strided_lengths = lengths.as_strided(size, stride)
        # Replicate the segment_reduce call from the example
        return torch.ops.aten.segment_reduce(
            a,
            "max",
            axis=0,
            unsafe=False,
            initial=1,
            lengths=strided_lengths
        )

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(5)
    lengths = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    return (a, lengths)

