import torch
import torch.nn as nn

from torch import nn
from torch._inductor import config

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10, device='cuda')
    
    def forward(self, x):
        return self.linear(x)
    

with torch.no_grad(), config.patch({"cpp_wrapper": True}):
    model = Model()
    model_opt = torch.compile(model)
    model_opt(torch.zeros(10, device="cuda"))