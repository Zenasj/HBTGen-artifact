py
import torch

dtype = torch.float64
device = 'cpu'
pin_memory = False
layout = None

x = torch.tensor(-4.8270, dtype=torch.float64)
size = (2,)
stride = ()


res = x.new_empty_strided(
    size,
    stride,
    dtype=dtype,
    device=device,
    pin_memory=pin_memory,
    layout=layout,
)

print(res.stride())
print(res)