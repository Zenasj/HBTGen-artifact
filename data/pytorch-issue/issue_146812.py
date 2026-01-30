import torch

print(torch.__version__)
input = torch.rand((11, 15,3))
print("Running test with non empty tensor")
print("="*50)
print(torch.ops.aten._pdist_forward(input, p=2.0))
print("="*50)
print("Running test with empty tensor")
print("="*50)
input = torch.rand((11, 15, 0))
print(torch.ops.aten._pdist_forward(input, p=2.0))