import torch

s = torch.rand(size=(3, 4)).to("cuda")

s_new = torch.zeros(s.shape)

s_new[:, :-1] = s[:, :-1].matmul(s_new[:, :-1])

import torch

s = torch.rand(size=(3, 4)).to("cuda")

s_new = torch.zeros(s.shape)

print(s[:, :-1].matmul(s_new[:, :-1]))