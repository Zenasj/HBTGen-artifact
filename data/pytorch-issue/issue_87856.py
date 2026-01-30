import torch

x = torch.tensor([[1,2]])
for device in ('cpu', 'mps'):
    y = x.to(device)
    z = torch.stack((y[:,:1], y[:,-1:]), dim=-1)
    print(z)