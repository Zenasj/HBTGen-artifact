import torch
rnd = torch.tensor([[ 0.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
         -1.0000, -1.0000, -1.0000],
        [ 0.5000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
         -1.0000, -1.0000, -1.0000],
        [ 8.0000,  8.0000,  8.0000,  8.0000,  8.0000,  8.0000,  8.0000, -1.0000,
         -1.0000, -1.0000, -1.0000],
        [ 7.0000,  7.0000,  7.0000,  7.0000,  7.0000,  7.0000, -1.0000, -1.0000,
         -1.0000, -1.0000, -1.0000]])

print(rnd.max(dim=1))
print((rnd.cuda()).max(dim=1))
""" my results: 
# CPU
torch.return_types.max(
values=tensor([0.0000, 0.5000, 8.0000, 7.0000]),
indices=tensor([0, 0, 6, 5]))
#CUDA
torch.return_types.max(
values=tensor([0.0000, 0.5000, 8.0000, 7.0000], device='cuda:0'),
indices=tensor([0, 0, 0, 0], device='cuda:0'))
"""

print("argmax")
print(torch.argmax(rnd, dim=1))

print("argmax cuda")
print(torch.argmax(rnd.cuda(), dim=1))

""" argmax results
argmax
tensor([0, 0, 6, 5])
argmax cuda
tensor([0, 0, 6, 5], device='cuda:0')
"""