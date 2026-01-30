import torch

def matmul_func(input, other, out):
  torch.matmul(input, other, out=out)

opt_model = torch.compile(matmul_func)

inp1 = torch.randn(10, 3, 4)
mat1 = torch.randn(4)
out1 = torch.empty(10, 3)
opt_model(inp1, mat1, out1)
print(out1.shape)

inp2 = torch.randn(12, 4, 5)
mat2 = torch.randn(5)
out2 = torch.empty(12, 4)
opt_model(inp2, mat2, out2)