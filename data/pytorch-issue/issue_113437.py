import torch
print("======== [2,3,4,5] nchw ========")
q = torch.randn((2,3,4,5))
print(q.size(),q.stride())  # torch.Size([2, 3, 4, 5]) (60, 20, 5, 1)
z = q.view(2,3,20)
print(z.size(),z.stride())  # torch.Size([2, 3, 20]) (60, 20, 1)

print("======== [2,3,4,5] nhwc ========")
q = torch.randn((2, 3, 4, 5)).to(memory_format=torch.channels_last)
print(q.size(),q.stride())   # torch.Size([2, 3, 4, 5]) (60, 1, 15, 3)
z = q.view(2,3,20)
print(z.size(),z.stride())   # torch.Size([2, 3, 20]) (60, 1, 3)

print("======== [1,3,4,5] nchw ========")
q = torch.randn((1, 3, 4, 5))
print(q.size(),q.stride())   # torch.Size([1, 3, 4, 5]) (60, 20, 5, 1)
z = q.view(1,3,20)
print(z.size(), z.stride())  # torch.Size([1, 3, 20]) (60, 20, 1)

print("======== [1,3,4,5] nhwc ========")
q = torch.randn((1, 3, 4, 5)).to(memory_format=torch.channels_last)
print(q.size(),q.stride())   # torch.Size([1, 3, 4, 5]) (60, 1, 15, 3)
z = q.view(1,3,20)
print(z.size(), z.stride())  # torch.Size([1, 3, 20]) (3, 1, 3)