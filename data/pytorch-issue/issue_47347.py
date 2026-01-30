import torch

# GPU
torch.manual_seed(0)
print(torch.distributions.Uniform(torch.tensor(0., device='cuda'), torch.tensor(1., device='cuda')).rsample((5,)))
# CPU
torch.manual_seed(0)
print(torch.distributions.Uniform(torch.tensor(0., device='cpu'), torch.tensor(1., device='cpu')).rsample((5,)))
# TPU
xm.set_rng_state(0)
print(torch.distributions.Uniform(torch.tensor(0., device=xm.xla_device()), torch.tensor(1., device=xm.xla_device())).rsample((5,)))