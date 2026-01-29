from typing import NamedTuple
import torch
import torch.nn as nn

class MyTuple(NamedTuple):
    a: torch.Tensor
    b: torch.Tensor

    def __getitem__(self, index):
        return MyTuple(self.a[index], self.b[index])

# torch.rand(2, dtype=torch.float32) for each element in the input tuple
class MyModel(nn.Module):
    def forward(self, my_tuple):
        try:
            result_attr = my_tuple.a + 1
        except:
            result_attr = None
        try:
            result_item = my_tuple[0] + 1
        except:
            result_item = None
        if result_attr is None or result_item is None:
            return torch.tensor(False, dtype=torch.bool)
        else:
            return torch.allclose(result_attr, result_item)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(2, dtype=torch.float32)
    b = torch.rand(2, dtype=torch.float32)
    return MyTuple(a, b)

