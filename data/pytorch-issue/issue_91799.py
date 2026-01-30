@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 754974720 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
...