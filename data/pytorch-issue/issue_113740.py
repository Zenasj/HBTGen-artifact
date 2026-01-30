# buf1709.size()=torch.Size([10, 80, 1, 30]) buf1708.size()=torch.Size([480, 80, 1, 1])
buf1711 = aten.convolution(buf1709, buf1708, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)

import torch
import itertools

aten = torch.ops.aten

with torch._subclasses.fake_utils.CrossRefFakeMode():
    for dtype, mem_fmt in itertools.product([torch.float, torch.half, torch.bfloat16], [torch.contiguous_format, torch.channels_last]):
        buf1709 = torch.rand([10, 80, 1, 30], device="cuda").to(dtype).to(memory_format=mem_fmt)
        buf1708 = torch.rand([480, 80, 1, 1], device="cuda").to(dtype).to(memory_format=mem_fmt)
        buf1711 = aten.convolution(buf1709, buf1708, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)