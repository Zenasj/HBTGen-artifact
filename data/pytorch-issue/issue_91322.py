import torch.nn as nn

# example of where 
import torch

t = torch.rand((4, 0, 0))
print("~")
print(torch.nn.functional.softmax(t, dim=-1))  # this passes
print("~")
torch._refs.softmax(t, dim=-1)  # this fails
print("~")