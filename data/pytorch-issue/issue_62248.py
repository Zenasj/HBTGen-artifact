import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim_factor=1.):
        super(InvertedResidual, self).__init__()

        hidden_dim = int(in_channels * hidden_dim_factor)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x

net = InvertedResidual(32, 32)
net.eval()
net_traced = torch.jit.trace(net, torch.randn(1,32,56,56))
net_traced.save("model.pth")
# net_traced_optimized = optimize_for_mobile(net_traced)
# net_traced_optimized.save("model.pth")