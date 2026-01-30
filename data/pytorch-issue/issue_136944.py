import torch
import torch.nn as nn

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.maxpool_op = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False).to(memory_format=torch.channels_last)

    def forward(self, input_tensor1, input_tensor2):
        output_pool = self.maxpool_op(input_tensor1)
        output_add = torch.add(output_pool, input_tensor2)
        return output_add

net = MyNetwork().cuda()
a = torch.rand([256, 64, 112, 112], device='cuda', dtype=torch.float16, requires_grad=True).to(memory_format=torch.channels_last)
b = torch.rand(256, 64, 56, 56, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last)
target_tensor = torch.randn(256, 64, 56, 56, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last)
net = torch.compile(net, backend='inductor', mode='max-autotune-no-cudagraphs').to(memory_format=torch.channels_last)
output_tensor = net(a, b)
print(output_tensor.sum())

criterion = torch.nn.MSELoss()
loss = criterion(output_tensor, target_tensor)

net.zero_grad() 
loss.backward() 
print(loss)

# kernel path: /tmp/torchinductor_root/qo/cqoyfr2ha6b6p5tmrykkgfbi2bm62zm7pubfqks4jrgs6hnptgvg.py
# Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]

triton_poi_fused_max_pool2d_with_indices_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_backward_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': None},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3211264
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112) % 112
    y2 = (yindex // 12544)
    y4 = yindex % 12544
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + (64*(tl.minimum(tl.maximum(0, (y0 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2)))))) + (64*(tl.where((tl.minimum(tl.maximum(0, (y0 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) >= 0, 0, 56))) + (3584*(tl.minimum(tl.maximum(0, (y1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2)))))) + (3584*(tl.where((tl.minimum(tl.maximum(0, (y1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2))))) >= 0, 0, 56))) + (200704*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((56*(tl.minimum(tl.maximum(0, (y1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2)))))) + (56*(tl.where((tl.minimum(tl.maximum(0, (y1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2))))) >= 0, 0, 56))) + (3136*x3) + (200704*y2) + (tl.minimum(tl.maximum(0, (y0 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) + (tl.where((tl.minimum(tl.maximum(0, (y0 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) >= 0, 0, 56))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr0 + (x3 + (64*(tl.minimum(1 + (tl.maximum(0, (y0 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2)))))) + (64*(tl.where((tl.minimum(1 + (tl.maximum(0, (y0 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) >= 0, 0, 56))) + (3584*(tl.minimum(tl.maximum(0, (y1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2)))))) + (3584*(tl.where((tl.minimum(tl.maximum(0, (y1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2))))) >= 0, 0, 56))) + (200704*y2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + ((56*(tl.minimum(tl.maximum(0, (y1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2)))))) + (56*(tl.where((tl.minimum(tl.maximum(0, (y1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2))))) >= 0, 0, 56))) + (3136*x3) + (200704*y2) + (tl.minimum(1 + (tl.maximum(0, (y0 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) + (tl.where((tl.minimum(1 + (tl.maximum(0, (y0 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) >= 0, 0, 56))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (x3 + (64*(tl.minimum(tl.maximum(0, (y0 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2)))))) + (64*(tl.where((tl.minimum(tl.maximum(0, (y0 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) >= 0, 0, 56))) + (3584*(tl.minimum(1 + (tl.maximum(0, (y1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2)))))) + (3584*(tl.where((tl.minimum(1 + (tl.maximum(0, (y1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2))))) >= 0, 0, 56))) + (200704*y2)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((56*(tl.minimum(1 + (tl.maximum(0, (y1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2)))))) + (56*(tl.where((tl.minimum(1 + (tl.maximum(0, (y1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2))))) >= 0, 0, 56))) + (3136*x3) + (200704*y2) + (tl.minimum(tl.maximum(0, (y0 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) + (tl.where((tl.minimum(tl.maximum(0, (y0 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) >= 0, 0, 56))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp30 = tl.load(in_ptr0 + (x3 + (64*(tl.minimum(1 + (tl.maximum(0, (y0 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2)))))) + (64*(tl.where((tl.minimum(1 + (tl.maximum(0, (y0 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) >= 0, 0, 56))) + (3584*(tl.minimum(1 + (tl.maximum(0, (y1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2)))))) + (3584*(tl.where((tl.minimum(1 + (tl.maximum(0, (y1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2))))) >= 0, 0, 56))) + (200704*y2)), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + ((56*(tl.minimum(1 + (tl.maximum(0, (y1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2)))))) + (56*(tl.where((tl.minimum(1 + (tl.maximum(0, (y1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y1) // 2))))) >= 0, 0, 56))) + (3136*x3) + (200704*y2) + (tl.minimum(1 + (tl.maximum(0, (y0 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) + (tl.where((tl.minimum(1 + (tl.maximum(0, (y0 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + y0) // 2))))) >= 0, 0, 56))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = y4
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.maximum(0, (y1 // 2))
    tmp10 = tl.minimum(56, 1 + ((1 + y1) // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.maximum(0, (y0 // 2)))
    tmp13 = tl.minimum(56, 1 + ((1 + y0) // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.maximum(0, (y1 // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.maximum(0, (y0 // 2))
    tmp25 = tmp24 < tmp13
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp21
    tmp28 = tmp18 + tmp20
    tmp29 = tl.where(tmp27, tmp28, tmp18)
    tmp32 = tmp30 == tmp2
    tmp33 = tmp23 & tmp14
    tmp34 = tmp33 & tmp32
    tmp35 = tmp29 + tmp31
    tmp36 = tl.where(tmp34, tmp35, tmp29)
    tl.store(out_ptr0 + (x3 + (64*y6)), tmp36, xmask)
''', device_str='cuda')