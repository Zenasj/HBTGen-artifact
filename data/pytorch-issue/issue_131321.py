# The input is a tuple of three tensors with shapes (4,), (6,), (6,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.compiled_scatter = torch.compile(torch.scatter_reduce)
        self.normal_scatter = torch.scatter_reduce

    def forward(self, inputs):
        input, index, src = inputs
        compiled_out = None
        normal_out = None

        try:
            compiled_out = self.compiled_scatter(input, 0, index, src, "sum")
        except:
            pass

        try:
            normal_out = self.normal_scatter(input, 0, index, src, "sum")
        except:
            pass

        # Compare outputs accounting for exceptions
        if compiled_out is None and normal_out is None:
            return torch.tensor(False)  # both failed
        elif compiled_out is None or normal_out is None:
            return torch.tensor(True)  # one failed, the other succeeded
        else:
            return torch.allclose(compiled_out, normal_out).unsqueeze(0)

def my_model_function():
    return MyModel()

def GetInput():
    input = torch.tensor([1., 2., 3., 4.])
    index = torch.tensor([0, 1, 0, 1, 2, -1])
    src = torch.tensor([1., 2., 3., 4., 5., 6.])
    return (input, index, src)

