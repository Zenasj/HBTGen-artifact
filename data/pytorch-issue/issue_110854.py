# torch.rand((), dtype=torch.uint8)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Compute incorrect result in uint8 (original bug scenario)
        incorrect = torch.square(x)
        # Compute correct result in int32 to avoid overflow
        correct = torch.square(x.to(torch.int32))
        # Compare by casting incorrect to int32 and checking difference
        diff = correct - incorrect.to(torch.int32)
        return diff != 0  # Returns True where overflow occurred

def my_model_function():
    return MyModel()

def GetInput():
    # Replicates original test case's input distribution
    return torch.randint(0, 100, (), dtype=torch.uint8)

