def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1028 # dynamic_shapes=False
    rnumel = 1028 # dynamic_shapes=False
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    _tmp1_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1028*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1_next, _tmp1_index_next = triton_helpers.minimum_with_index(
            _tmp1, _tmp1_index, tmp0, rindex
        )
        _tmp1 = tl.where(rmask & xmask, _tmp1_next, _tmp1)
        _tmp1_index = tl.where(rmask & xmask, _tmp1_index_next, _tmp1_index)
    _, tmp1_tmp = triton_helpers.min_with_index(_tmp1, _tmp1_index, 1)
    tmp1 = tmp1_tmp[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)