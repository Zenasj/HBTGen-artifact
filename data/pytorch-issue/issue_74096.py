# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class A:
    def __init__(self):
        super().__init__()
        self.a = True

class B(A, nn.Module):
    def __init__(self):
        super().__init__()
        self.b = True

class C(nn.Module, A):
    def __init__(self):
        super().__init__()
        self.c = True

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b_sub = B()  # Submodule B (A, nn.Module)
        self.c_sub = C()  # Submodule C (nn.Module, A)

    def forward(self, x):
        # Compare presence of 'a' in both submodules
        has_a_b = hasattr(self.b_sub, 'a')
        has_a_c = hasattr(self.c_sub, 'a')
        return torch.tensor([has_a_b and has_a_c], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

