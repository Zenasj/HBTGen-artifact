import torch
results = dict()

input = torch.rand([1, 1, 4], dtype=torch.float64, requires_grad=True)
other = torch.rand([4], dtype=torch.float16, requires_grad=True)

res = torch.xlogy(input, other)

sum_res = res.sum()
sum_res.backward()
# RuntimeError: expected scalar type double but found c10::Half