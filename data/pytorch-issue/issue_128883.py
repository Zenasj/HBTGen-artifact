@torch.compile
def fn(x, y):
    x = x.view(torch.float16)
    y = y.view(torch.float16) + 1
    return x @ y

x = torch.randn((2, 2), device=self.device, dtype=torch.bfloat16)
y = torch.randn((2, 2), device=self.device, dtype=torch.bfloat16)
fn(x, y)

import torch
@torch.compile
def fn6(x):
    x = x + 1
    x = torch.ops.aten.view.dtype(x, torch.int16)
    x = x * 2
    return x

x = torch.randn([1024], dtype=torch.float16, device="cuda")
fn6(x)

with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, ), (1, ), torch.int16)
        # Source Nodes: [x_2], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, buf0, 1024, grid=grid(1024), stream=stream0)
        del arg0_1