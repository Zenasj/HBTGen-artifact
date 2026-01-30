import torch
import torch.nn as nn

class Foo(torch.nn.Module):
    def forward(self, x, y):
        x = x[x > 0]
        y = y[y > 0]
        return max(x.shape[0], y.shape[0])

model = Foo()
inputs = (torch.randn(64), torch.randn(64))
with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
    ep = export(model, inputs, strict=False)