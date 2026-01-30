import torch
import torch.nn as nn

def test_group_norm_backward(device="cuda"):
    B,C,W,H=2,4,4,4
    net=torch.nn.GroupNorm(B, C).to(device=device)
    x=torch.rand(B, C, W, H, device=device, requires_grad=True)
    y=net(x)
    y.backward(torch.rand(B, C, W, H, device=device).to(memory_format=torch.channels_last))