import torch.nn as nn

import torch

dtype = torch.half

for i in range(0,500):
    torch.cuda.manual_seed(17)
    # dummy tensors to fuzz where ConvTranspose2d inputs get allocated
    dummy0 = torch.empty((7777*i,), device="cuda", dtype=dtype)
    a = torch.randn((6, 256, 50, 50), device="cuda", dtype=dtype)
    dummy1 = torch.empty((7777*i,), device="cuda", dtype=dtype)
    # shape from CycleGAN
    m = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True).cuda().to(dtype)
    dummy2 = torch.empty((7777*i,), device="cuda", dtype=dtype)
    out = m(a)
    out.sum().backward()
    print(i,
          "weight grad sum = ", m.weight.grad.data.double().sum().item() if m.weight.grad is not None else None,
          "bias grad sum = ", m.bias.grad.data.double().sum().item() if m.bias.grad is not None else None)
    m.zero_grad()