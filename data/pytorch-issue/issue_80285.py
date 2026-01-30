import torch

t = torch.uint8

t0_cpu = torch.arange(4*3, dtype=t).reshape((4, 3))
t0_mps = t0_cpu.to(torch.device('mps'))

t1_cpu = t0_cpu.permute((1, 0))
t1_mps = t0_mps.permute((1, 0))

print('CPU')
print(' in:')
print(t0_cpu)
print(' out:')
print(t1_cpu)

print('MPS')
print(' in:')
print(t0_mps)
print(' out:')
print(t1_mps)