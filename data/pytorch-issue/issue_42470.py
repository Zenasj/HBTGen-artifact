import torch
import torch.nn as nn

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Softshrink(),
    torch.nn.Linear(H, D_out),
)