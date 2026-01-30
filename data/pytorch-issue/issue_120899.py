import torch

t_mps = torch.tensor([torch.nan, 1, 2], device="mps")
print(torch.clamp(t_mps, min=-100, max=100))
# tensor([-100.,    1.,    2.], device='mps:0')
print(torch.clamp(t_mps, min=-100))
# tensor([-100.,    1.,    2.], device='mps:0')
print(torch.clamp(t_mps, max=100))
# tensor([100.,   1.,   2.], device='mps:0')

t_cpu = torch.tensor([torch.nan, 1, 2], device="cpu")
print(torch.clamp(t_cpu, min=-100, max=100))
# tensor([nan, 1., 2.])
print(torch.clamp(t_cpu, min=-100))
# tensor([nan, 1., 2.])
print(torch.clamp(t_cpu, max=100))
# tensor([nan, 1., 2.])

t_mps = torch.tensor([torch.nan, 1, 2], device="mps")
t_mps[t_mps < -100] = -100
t_mps[t_mps > 100] = 100
print(t_mps)
# tensor([nan, 1., 2.], device='mps:0')