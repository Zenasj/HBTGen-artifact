import torch

def topk_func(input, k, out):
  torch.topk(input, k, out=out)

values = torch.empty(3)
indices = torch.empty(3, dtype=torch.long)
opt_model = torch.compile(topk_func)

x = torch.arange(1., 6.)
opt_model(x, 3, out=(values, indices))

x = torch.arange(1., 8.)
opt_model(x, 3, out=(values, indices))

def bmm_func(input, mat, out):
  torch.bmm(input, mat, out=out)

opt_model = torch.compile(bmm_func)

inp1 = torch.randn(10, 3, 4)
mat1 = torch.randn(10, 4, 5)
out1 = torch.empty(10, 3, 5)
opt_model(inp1, mat1, out1)

inp2 = torch.randn(12, 4, 5)
mat2 = torch.randn(12, 5, 6)
out2 = torch.empty(12, 4, 6)
opt_model(inp2, mat2, out2)

def max_func(input, out):
  torch.max(input, 0, keepdim=True, out=out)

max = torch.empty(1)
max_indices = torch.empty(1, dtype=torch.long)

opt_model = torch.compile(max_func)

inp1 = torch.randn(4)
opt_model(inp1, out=(max, max_indices))

inp2 = torch.randn(5)
opt_model(inp2, out=(max, max_indices))