import torch
import random
from torch import nn

# torch.rand(1, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        seed = random.randint(0, 100)
        rand = random.Random(seed)
        return x + rand.randrange(10)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)

