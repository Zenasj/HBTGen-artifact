import torch
x=torch.randn(10,8, dtype=torch.complex64).t()[::2, ::2]
print(x.clone().stride())