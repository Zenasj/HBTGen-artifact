import torch.nn as nn

import torch
import numpy as np


torch_x = torch.tensor(np.load("./input.npy"), device="cuda")
torch_x.requires_grad=True
m2 = torch.nn.Conv3d(720, 720, (2, 2, 2), stride=(4, 4, 1), padding=(1, 1, 1), groups=3)
m2.weight.data = torch.tensor(np.load("./weight.npy"))
m2.bias.data = torch.tensor(np.load("./bias.npy"))
m2.to("cuda")
y2 = m2(torch_x)
y2.sum().backward()


torch_x_cpu = torch.tensor(np.load("./input.npy"))
torch_x_cpu.requires_grad=True
m2_cpu = torch.nn.Conv3d(720, 720, (2, 2, 2), stride=(4, 4, 1), padding=(1, 1, 1), groups=3)
m2_cpu.weight.data = torch.tensor(np.load("./weight.npy"))
m2_cpu.bias.data = torch.tensor(np.load("./bias.npy"))
y2_cpu = m2_cpu(torch_x_cpu)
y2_cpu.sum().backward()

assert(np.allclose(m2.weight.grad.cpu().numpy().flatten(), m2_cpu.weight.grad.cpu().numpy().flatten(), 1e-3, 1e-3))

import torch
import numpy as np

torch.manual_seed(1)

state_dict_cpu = {
    'weight': torch.randn(720, 240, 2, 2, 2),
    'bias': torch.randn(720)
}
state_dict_gpu = {k: v.to('cuda') for k, v in state_dict_cpu.items()}

m_cpu = torch.nn.Conv3d(720, 720, (2, 2, 2), stride=(4, 4, 1), padding=(1, 1, 1), groups=3)
m_gpu = torch.nn.Conv3d(720, 720, (2, 2, 2), stride=(4, 4, 1), padding=(1, 1, 1), groups=3, device='cuda')
m_cpu.load_state_dict(state_dict_cpu)
m_gpu.load_state_dict(state_dict_gpu)

# Disabling mkldnn fixes the problem.
#with torch.backends.mkldnn.flags(enabled=False):

input_cpu = torch.randn(1, 720, 4, 4, 6)
input_gpu = input_cpu.to('cuda')

y_cpu = m_cpu(input_cpu)
y_cpu.sum().backward()
y_gpu = m_gpu(input_gpu)
y_gpu.sum().backward()

grad_cpu = m_cpu.weight.grad
grad_gpu = m_gpu.weight.grad.cpu()
print('grad_cpu max:', grad_cpu.max())
print('grad_gpu max:', grad_gpu.max())
assert(torch.allclose(grad_cpu, grad_gpu, 1e-3, 1e-3))