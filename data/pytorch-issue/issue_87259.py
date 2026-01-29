# torch.rand(100000, dtype=torch.float32)  # Input is a 1D tensor of size 100k
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.serial_sum = SerialSum()
        self.parallel_sum = ParallelSum()  # Will use parallel reduction when compiled

    def forward(self, x):
        s_serial = self.serial_sum(x)
        s_parallel = self.parallel_sum(x)
        # Return True if difference exceeds tolerance (1e-6) to mimic test failure criteria
        return torch.abs(s_serial - s_parallel) > 1e-6

class SerialSum(nn.Module):
    def forward(self, x):
        total = x.new_zeros(())
        for val in x.flatten():
            total += val  # Serial accumulation in memory order
        return total

class ParallelSum(nn.Module):
    def forward(self, x):
        return x.sum()  # May use parallel reduction when compiled

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100000, dtype=torch.float32)  # Matches input shape in the example

