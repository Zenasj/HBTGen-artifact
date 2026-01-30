tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask)
tmp1 = tmp0.to(tl.int64)
tmp2 = (tmp1 != 0)

tmp5 = tl.where(rmask & xmask, tmp3, 0)

tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask).to(tl.int1)
tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
tmp3 = tl.full([1, 1], 0, tl.int1)
tmp4 = tl.where(rmask & xmask, tmp1, tmp3)
tmp5 = triton_helpers.any(tmp4, 1)[:, None]