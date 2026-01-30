import torch
import torch.nn as nn

class Baz(torch.nn.Module):
    def forward(self, x):
        x = torch.where(x <= 0.5)[0]
        y = torch.randn(x.shape[0], 4)
        if y.numel() < 200:
            return x + y[:, 0]
model = Baz()
inputs = (torch.randn(64, 32),)

with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
    ep = export(model, inputs, strict=False)

class Baz(torch.nn.Module):
    def forward(self, x):
        x = torch.where(x <= 0.5)[0]
        if x.shape[0] < 200:
            return x + 2
model = Baz()
inputs = (torch.randn(64, 32),)

with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
    ep = export(model, inputs, strict=False)