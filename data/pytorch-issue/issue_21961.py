import torch.nn as nn
import numpy as np

import torch

loss_fn = torch.nn.CTCLoss(reduction='none')
inp = torch.randn(10, 2, 10)
tar = torch.randint(10, (4,))
inp_len = torch.full((2,), 10, dtype=torch.long)
tar_len = torch.full((2,), 2, dtype=torch.long)

inp_cpu = inp.clone().detach()
inp_cpu.requires_grad = True
loss_fn(inp_cpu, tar, inp_len, tar_len).sum().backward()
print(inp_cpu.grad.abs().sum().item())

inp_cudnn = inp.cuda().detach()
inp_cudnn.requires_grad = True
loss_fn(inp_cudnn, tar.cuda(), inp_len.cuda(), tar_len.cuda()).sum().backward()
print(inp_cpu.grad.abs().sum().item())

inp_cudnn = inp.cuda().detach()
inp_cudnn.requires_grad = True
loss_fn(inp_cudnn, tar.cuda(), inp_len.cuda(), tar_len.cuda()).sum().backward()
print(inp_cpu.grad.abs().sum().item())

inp_cpu = inp.clone().detach()
inp_cpu.requires_grad = True
loss_fn(inp_cpu, tar, inp_len, tar_len).mean().backward()
print(inp_cpu.grad.abs().sum().item())

loss = (loss_by_sample * sample_weight).sum()
loss.backward()

inp_cudnn = inp.cuda().detach()
inp_cudnn.requires_grad = True
loss_fn(inp_cudnn, tar.cuda(), inp_len.cuda(), tar_len.cuda()).mean().backward()
print(inp_cpu.grad.abs().sum().item())

inp_cpu = inp.clone().detach()
inp_cpu.requires_grad = True
loss_fn(inp_cpu, tar, inp_len, tar_len).mean().backward()
print(inp_cpu.grad.abs().sum().item())