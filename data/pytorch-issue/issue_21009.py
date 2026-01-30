import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit


@torch.jit.script
def resize_ref(x, shortpoint, method: str='bilinear', align_corners: bool=True):
    """
    :type x: torch.Tensor
    :type shortpoint: torch.Tensor
    :type method: str
    :type align_corners: bool
    """
    hw = shortpoint.shape[2:4]
    ihw = x.shape[2:4]
    if hw != ihw:
        x = F.interpolate(x, hw, mode=method, align_corners=align_corners)
    return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x2 = F.interpolate(x, scale_factor=2)
        y = resize_ref(x, x2)
        return y


a = torch.rand(1, 3, 6, 6)
net = Net()
net = torch.jit.trace(net, a)
b = net(a)
print(b.shape)