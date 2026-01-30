import torch
import torch.nn as nn
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(10, 5)
        self.ln = nn.LayerNorm(5, elementwise_affine=True)
  
    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        return x

x = torch.randn(2, 10)
m = MyModel()
m.fc.weight.requires_grad = False
m.fc.bias.requires_grad = False

out = call_for_per_sample_grads(m, x.shape[0], x)
out.sum().backward()