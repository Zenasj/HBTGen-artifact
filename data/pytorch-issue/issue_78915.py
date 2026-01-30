import torch
xs = torch.arange(30).to('mps')
print(xs.topk(30))

torch.return_types.topk(
values=tensor([29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], device='mps:0'),
indices=tensor([29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], device='mps:0'))