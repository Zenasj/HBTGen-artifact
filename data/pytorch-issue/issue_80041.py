import torch
import torch.nn as nn

a = torch.randn(64,64)
b = torch.randn(64,64).T

torch.manual_seed(42)
torch.nn.init.uniform_(a)
torch.manual_seed(42)
torch.nn.init.uniform_(b)

(a==b).all()  # False.