import torch
tmp = torch.randn(512, 768)

tmp_delta = torch.diff(tmp, dim=0, prepend=torch.zeros(1,768)) #prepend zeros so that the first entry is preserved

cumsum_delta = torch.cumsum(tmp_delta, dim=0)

print(torch.allclose(tmp, cumsum_delta))# False, but should be true
print(tmp == cumsum_delta)

print(torch.allclose(tmp, cumsum_delta, atol=1e-5, rtol=1e-5))  # True