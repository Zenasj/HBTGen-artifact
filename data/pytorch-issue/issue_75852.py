import torch

params = torch.rand(1, requires_grad=True)
out = torch.stack([params, 1j*params], 1)
loss = out.sum()
print(loss)
loss.backward()

out = torch.stack([params.cfloat(), 1j*params], 1)

out = torch.cat([params.unsqueeze(1), 1j*params.unsqueeze(1)], 1)