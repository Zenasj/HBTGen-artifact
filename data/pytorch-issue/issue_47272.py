import torch

device = torch.device('cuda:0')

d = torch.tensor([
    [[-0.4397,  0.8981,  0.0000], [-0.8981, -0.4397,  0.0000], [ 0.0000,  0.0000,  1.0000]],
    [[-0.4397,  0.8981,  0.0000], [-0.8981, -0.4397,  0.0000], [ 0.0000,  0.0000,  1.0000]]
], device=device)

count = 0
while True:
    temp = torch.inverse(d)
    count = count + 1
    assert not torch.isnan(temp).any(), str(temp) + '\n' + str(count)