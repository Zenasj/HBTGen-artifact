import torch
import torch.nn as nn

cuda = 'cuda:0'
with torch.no_grad():
    cpu = torch.rand(1, 32, 256, 256, 72, device='cpu')
    gpu = cpu.to(cuda)
    m = nn.GroupNorm(4, 32).eval()
    cpu = m.cpu()(cpu)
    gpu = m.to(cuda)(gpu)
    print(torch.abs(cpu - gpu.cpu()).max())  # 0.0121 on my machine!!