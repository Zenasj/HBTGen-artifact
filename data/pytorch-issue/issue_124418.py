import torch
from torch.testing._internal.two_tensor import TwoTensor

t = TwoTensor(torch.randn(2, 2, device='cuda'), torch.randn(2, 2, device='cuda'))
print(t.device, t.a.device, t.b.device)  # cuda:0 cuda:0 cuda:0
torch.save(t, 'subclass.pt')
t_loaded = torch.load('subclass.pt')
print(t_loaded.device, t_loaded.a.device, t_loaded.b.device)  # cuda:0 cuda:0 cuda:0
t_loaded_cpu = torch.load('subclass.pt', map_location='cpu')
print(t_loaded_cpu.device, t_loaded_cpu.a.device, t_loaded_cpu.b.device)  # cuda:0 cpu cpu