import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

# torch.rand(1, dtype=torch.float32)
V = np.random.rand(17, 17)
scipy.linalg.inv(V)

class MyModel(nn.Module):
    @staticmethod
    @torch.jit.script
    def _scripted_subroutine(x):
        return x + 0  # Dummy computation to trigger TorchScript

    def forward(self, x):
        return self._scripted_subroutine(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

