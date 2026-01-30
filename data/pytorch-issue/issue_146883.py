import torch
from torch._dynamo.testing import rand_strided
assert_size_stride = torch._C._dynamo.guards.assert_size_stride

devices = ["cpu", "xpu"]

for device in devices:
    print("testing device: ", device)
    arg0_1 = rand_strided((32, 4), (4, 1), device=device, dtype=torch.float32)
    buf0 = torch.ops.aten.nonzero.default(arg0_1)
    assert_size_stride(buf0, (128, 2), (1, 128))