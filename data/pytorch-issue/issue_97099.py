@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks0) % ks0
    x0 = xindex % ks0
    x2 = (xindex // ks3)
    x4 = xindex
    tmp0 = x1 + ((-1)*ks1)
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = x0 + ((-1)*ks1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (x0 + ((-1)*ks1) + (ks2*x1) + (x2*(ks2*ks2)) + ((-1)*ks1*ks2) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp12, xmask)