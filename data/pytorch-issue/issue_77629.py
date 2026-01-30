import torch
input = torch.rand([4, 3], dtype=torch.float32, requires_grad=True)
other = torch.rand([1], dtype=torch.float32, requires_grad=True)

res = torch.cross(input, other)
print("forward pass")
res.sum().backward()
# forward pass
# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)