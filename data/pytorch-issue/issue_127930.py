tmp6 = tl.load(in_ptr0 + (tl.where(((-2) + x0) < 0, 3 + x0, (-2) + x0)), tmp5 & xmask, other=0.0)
tmp7 = tl.full(tmp6.shape, 7, tmp6.dtype)
tmp8 = tl.where(tmp5, tmp6, tmp7)

tmp8 = tl.load(in_ptr0 + (tl.where(((-2) + x0) < 0, 3 + x0, (-2) + x0)), tmp5 & xmask, other=7.0)