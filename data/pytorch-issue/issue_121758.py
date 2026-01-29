# torch.rand(4, 6, dtype=torch.float), torch.tensor([0, 2, 4], dtype=torch.int64)
import torch
from torch import nn
from torch.nested._internal.nested_tensor import ViewNestedFromBuffer, buffer_from_jagged

class MyModel(nn.Module):
    def __init__(self, use_nt=True):
        super().__init__()
        self.l = nn.Linear(6, 8)
        self.use_nt = use_nt

    def forward(self, inputs):
        values, offsets = inputs
        if self.use_nt:
            nt = ViewNestedFromBuffer.apply(values, offsets)
            output_nt = self.l(nt)
            return buffer_from_jagged(output_nt)
        else:
            return self.l(values)

def my_model_function():
    # Returns the model using NestedTensor (as in the original repro's failing case)
    return MyModel(use_nt=True)

def GetInput():
    values = torch.rand(4, 6, dtype=torch.float, requires_grad=True)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    return (values, offsets)

