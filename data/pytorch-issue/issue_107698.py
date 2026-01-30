import torch

input = torch.tensor([[1, 2], [3, 4]], dtype=torch.int)
indices = [torch.tensor([[True, False], [False, True]], dtype=torch.bool)]

# Both should return tensor [1, 4], but it seems index doesn't properly handle dtype conversions
print(torch.ops.aten.index.Tensor_out(input, indices, out=torch.tensor([0, 0], dtype=torch.int)))
print(torch.ops.aten.index.Tensor_out(input, indices, out=torch.tensor([0, 0], dtype=torch.long)))

import torch
torch.add(torch.rand(3), torch.rand(3),out=torch.empty(3,dtype=torch.float16))
# tensor([0.8267, 0.2036, 1.7734], dtype=torch.float16)

import torch
x=torch.arange(4, dtype=torch.int).resize(2, 2)
x[torch.tensor([[True, False], [False, True]])]=torch.empty(2, dtype=torch.float)
# Raises Index put requires the source and destination dtypes match