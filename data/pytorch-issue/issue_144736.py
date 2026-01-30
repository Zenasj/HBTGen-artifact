import torch
from torch._subclasses import FakeTensorMode

device = "cuda"

x = torch.randn((24, 16, 32, 32), device=device).to(memory_format=torch.channels_last)
x = x.view(2, 12, 16, 32, 32)

i1 = torch.arange(2).unsqueeze(-1)
i2 = torch.argsort(torch.rand(2, 12), dim=-1)[:, :3]

print(f"Eager stride: {x[i1, i2].stride()}")

mode = FakeTensorMode()
with mode:
    f_x = mode.from_tensor(x)
    f_i1 = mode.from_tensor(i1)
    f_i2 = mode.from_tensor(i2)
    f_out = f_x[f_i1, f_i2]
    print(f"Meta stride: {f_out.stride()}")