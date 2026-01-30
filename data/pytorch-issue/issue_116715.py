def call(args):
    primals_1, gt, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (1, 320, 128, 128), (5242880, 1, 40960, 320))
    assert_size_stride(gt, (1, 320, 128, 128), (5242880, 1, 40960, 320))
    assert_size_stride(tangents_1, (1, 320, 128, 128), (5242880, 16384, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty((1, 320, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_0.run(tangents_1, gt, primals_1, buf0, 320, 16384, grid=grid(320, 16384), stream=stream0)
        del gt
        del primals_1
        del tangents_1
        return (buf0, )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

class Block(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Dropout(p=0.1),
        )
        
    def forward(self, x):
        out = F.gelu(x)
        out = self.in_layers(out)

        return out

model = Block()
model = model.to("cuda").to(memory_format=torch.channels_last)

net = torch.compile(model, fullgraph=True)

num_batch = 20
profile_batch = 15

for i in range(num_batch):

    input_t = torch.randn([1, 320, 128, 128], dtype=torch.float32, device='cuda', requires_grad=True)
    input_t = input_t.to(memory_format=torch.channels_last)

    target_t = torch.ones_like(input_t)

    if i == profile_batch:
        torch.cuda.cudart().cudaProfilerStart()
    
    out = net(input_t)
    out.backward(target_t)

    if i == profile_batch:
        torch.cuda.cudart().cudaProfilerStop()