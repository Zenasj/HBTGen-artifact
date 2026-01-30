import torch # 2.5.1+cu124

src = torch.tensor([1., 2., 3., 4., 5., 6.])
index = torch.tensor([1, 1, 0, 1, 2, 1])
input = torch.tensor([1., 2., 3., 4.])

# Simulate a batch dimension of 1
src = src.unsqueeze(0)
index = index.unsqueeze(0)
input = input.unsqueeze(0)

def _fn(inputs):
    _src, _index, _input = inputs
    return torch.scatter_reduce(_input, 0, _index, _src, reduce="sum")

result = torch.vmap(_fn)((src, index, input))
print(result)