import numpy as np

import torch
from torch.utils.checkpoint import checkpoint

class ConvModel(torch.nn.Module):
    def __init__(self, n_in):
        super(ConvModel, self).__init__()
        self.init = torch.nn.Conv2d(n_in, n_in, 1)
        self.conv = torch.nn.Conv2d(n_in, n_in, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.init(x) # natural way to enable x.requires_grad before the checkpoint
        x = checkpoint(self.conv, x)
        return x


# Some constants
batch_size = 1
n_in = 10
h_w = 10

cuda = torch.cuda.is_available()

m = ConvModel(n_in)
if(cuda):
    m = m.cuda()
optimizer = torch.optim.Adam(m.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

# Simple training loop; nothing fancy here
for i in range(1000):
    optimizer.zero_grad()

    # Arbitrary inputs
    inp = torch.rand((batch_size, h_w, h_w, n_in))
    target = torch.zeros((batch_size, h_w, h_w), dtype=torch.int64)
    if(cuda):
        inp = inp.cuda()
        target = target.cuda()

    out = m(inp)
    loss = loss_fn(out, target)
    loss.backward() # crash occurs here
    optimizer.step()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


if __name__ == '__main__':
    device = torch.device('cuda')

    layernorm = LayerNorm2d(3).to(device)
    conv = nn.Conv2d(3, 3, kernel_size=1).to(device)

    x = torch.ones(4, 3, 224, 224, device=device, requires_grad=True)
    x = layernorm(x)
    x = checkpoint(conv, x)

    loss = torch.mean(x)
    loss.backward()