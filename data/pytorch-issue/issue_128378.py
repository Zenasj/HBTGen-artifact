import torch

a = ops.indirect_indexing(...)
b = ops.index_expr(a, ...)
c = ops.indirect_indexing(b, ...)

def forward(self, arg0_1, arg1_1, arg2_1):
    iota = torch.ops.prims.iota.default(arg0_1, start = 0, step = 1, index=0),
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(arg1_1);
    index = torch.ops.aten.index.Tensor(iota, [repeat_interleave]);
    index_1 = torch.ops.aten.index.Tensor(arg2_1, [index]);
    return (index_1,)

def triton_poi_fused_index_select_0(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    ...
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = ks0
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    # check_bounds()
    tl.device_assert(((0 <= tmp4) & (tmp4 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp4 < ks0")

def call():
  arg0_1, arg1_1, arg2_1 = args
  buf1 = aten.repeat_interleave.Tensor(arg1_1)
  buf4 = empty_strided_cuda((u0, 64), (64, 1))
  triton_poi_fused_index_select_0.run(
    buf1, arg2_1, buf4, s0, 
    triton_poi_fused_index_select_0_xnumel, 
    grid=grid(triton_poi_fused_index_select_0_xnumel), 
    stream=stream0)

tmp4 = tl.where(tmp3, tmp2, tmp0)
# One from codegen pass
tl.device_assert(((0 <= tmp4) & (tmp4 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp4 < ks0")
# One from IndexPropagation -- this one seems like the stricter bounds check (but safer?). 
tl.device_assert(((0 <= tmp4) & (tmp4 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 32")

# First check_bounds() from torch.arange(repeats.numel())
tl.device_assert(((0 <= tmp4) & (tmp4 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp4 < ks0")
# Second check_bounds() from torch.index_select(x ...)
tl.device_assert((tmp4 < 32) | ~(xmask), "index out of bounds: tmp4 < 32")