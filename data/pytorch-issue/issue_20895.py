import torch
a = torch.randn(2, 3).to_mkldnn()
c = torch.randn(2, 3).to_mkldnn()
b = torch.clone(a)
b.add_(c)
print(a)
print(b)