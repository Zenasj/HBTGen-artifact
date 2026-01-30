import torch 

N = 21

dt = torch.int8
# device = "cuda:0"
device = "cpu"

a = torch.ones(N, dtype=dt, device=device)
indices = torch.arange(0, N, dtype=torch.uint8, device=device)
values = torch.ones(N, dtype=dt, device=device)

print(a.shape)
print(indices.shape)
print(values.shape)

a.index_put_((indices, ), values, accumulate=True)

print(a)