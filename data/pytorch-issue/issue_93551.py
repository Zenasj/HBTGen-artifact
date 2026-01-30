import torch.nn as nn

import torch
from torch import nn
import torch._dynamo

torch.manual_seed(0)
torch.cuda.manual_seed(0)
import torch.profiler as profiler
torch._inductor.config.tune_layout = True


class Conv25d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.spatial_conv = nn.Conv2d(*args, **kwargs)
        args = (args[1], *args[1:])
        d1_kwargs = dict(kwargs)
        d1_kwargs['stride'] = 1
        self.temporal_conv = nn.Conv1d(*args, **d1_kwargs)
        torch.nn.init.eye_(self.temporal_conv.weight[:, :, (args[-1]-1)//2])
    def forward(self, input):
        B, C, F, H, W = input.size()
        x = input.permute(0, 2, 3, 4, 1)
        x = x.contiguous().view(-1, H, W, C).permute(0, 3, 1, 2)
        x = self.spatial_conv(x)
        x = x.view(B, F, -1, H, W)
        x = x.permute(0, 3, 4, 1, 2)
        out_channel = x.size(4)
        x = x.contiguous().view(-1, F, out_channel).permute(0, 2, 1)
        x = self.temporal_conv(x)
        x = x.view(B, H, W, -1, F)
        x = x.permute(0, 3, 4, 1, 2)
        return x

conv25d = Conv25d(320, 320, 3, padding=1).cuda().half()
x = torch.rand((4, 320, 16, 256, 256), dtype=torch.half, device='cuda')

@torch.no_grad()
@torch._dynamo.optimize('inductor')
def call_model(conv25d, input):
    return conv25d(input)

for i in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if i == 3:
        with profiler.profile() as prof:
            result = call_model(conv25d, x)
        prof.export_chrome_trace("profiler_conv25d.json")
    else:
        result = call_model(conv25d, x)

    end.record()
    torch.cuda.synchronize()

    print(start.elapsed_time(end))