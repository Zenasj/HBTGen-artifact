def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    # ...
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp2 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp3 = 100.0
        tmp4 = tmp1 / tmp3
        tmp5 = tmp2 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask, _tmp8 + tmp2, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    # ...