import torch.nn as nn

import torch
import torch.nn.functional as F

for eps in (0.000, 1e-8, 1e-7, 1e-6, 1e-5):
    input = torch.randn(300, 256, requires_grad=True, device='cuda')
    target = input.detach() + eps

    loss = F.smooth_l1_loss(input, target, beta=0.0, reduction='sum')
    loss.backward()

    if eps == 0:
        print(f"{eps:.3f}: {input.grad.mean().item()}")
    else:
        print(f"{eps}: {input.grad.mean().item()}")