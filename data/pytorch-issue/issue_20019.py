import torch
print(torch.__version__)
print(torch.prod(torch.ones(10 ** 6, device='cuda')))

print(torch.prod(torch.ones(130_560, device='cuda'), dim=0))
print(torch.prod(torch.ones(130_561, device='cuda'), dim=0))

tensor(1., device='cuda:0')
tensor(0., device='cuda:0')