import torch

torch.manual_seed(100)
torch.set_printoptions(precision=1, linewidth=150)

t0_cpu = (torch.rand((16,)) * 255).to(torch.uint8)
t0_mps = t0_cpu.to(torch.device('mps'))

t1_cpu = t0_cpu.to(torch.float32)
t1_mps = t0_mps.to(torch.float32)

print(f'CPU:\n{t0_cpu}\n{t1_cpu}\nMPS:\n{t0_mps}\n{t1_mps}')