import torch

input = torch.randn(10)
index = torch.randint(10, (20,))
value = torch.randn(20)

torch.ops.aten.index_put.hacked_twin(input, [index], value, accumulate=False)