import torch

a = torch.randn(1,2,3).cuda()
print(a)

tensor([[[ 0.9284, -0.6863,  1.5324],
         [-0.6943, -0.7309,  0.4542]]], device='cuda:0')