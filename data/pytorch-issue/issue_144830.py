# torch.rand(1, dtype=torch.int)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        num = x.item()
        # Method A (problematic): uses built-in format() function
        s_a = format(num, "b")
        # Method B (working): uses string's format() method
        s_b = "{:b}".format(num)
        # Compare outputs (returns True if both methods agree)
        return torch.tensor(s_a == s_b, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 100, (1,), dtype=torch.int)

