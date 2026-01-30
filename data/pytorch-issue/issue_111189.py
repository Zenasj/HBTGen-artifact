import torch
torch.set_num_threads(1)
x = torch.full((32768,), -1, dtype=torch.int32)
x[:100] = torch.iinfo(x.dtype).max
uv=x.sort().values.unique()
print(uv)
assert uv.size(0) == 2