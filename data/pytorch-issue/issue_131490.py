import torch
torch.set_future_lazy_clone(True)

cpu = torch.zeros((1, 2, 3), device=torch.device('cpu'))
mps = cpu.to("mps")

print(mps)