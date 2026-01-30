import torch
v = torch.randn((2, 3), device="cuda:0")
print(v)
print(v.to("cuda:1"))
print(v.to("cpu").to("cuda:1"))

tensor([[-1.2347,  0.9847, -0.6321],
        [-0.9209,  0.5088, -1.5387]], device='cuda:0')
tensor([[-0.6436, -0.7371, -0.5609],
        [-1.4677, -0.4079,  1.3875]], device='cuda:1')
tensor([[-1.2347,  0.9847, -0.6321],
        [-0.9209,  0.5088, -1.5387]], device='cuda:1')