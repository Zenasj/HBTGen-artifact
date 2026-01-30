import torch.nn as nn

import torch

CHANNELS = 4


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x, emb):
        x = self.conv(x) + emb  # (batch_size, 1, ...) + (batch_size, CHANNELS, ...)
        return x


net = MyModule()
net = net.cuda()

opt_net = torch.compile(net, dynamic=True)

batch_sizes = [7, 5, 8]
lengths = [297, 120, 361]
for batch_size, length in zip(batch_sizes, lengths):
    print(f'batch_size={batch_size}, length={length}')
    x = torch.randn(batch_size, 1, 256, length, device='cuda')
    emb = torch.randn(batch_size, CHANNELS, 1, 1, device='cuda')
    out = opt_net(x, emb)
    out.mean().backward()