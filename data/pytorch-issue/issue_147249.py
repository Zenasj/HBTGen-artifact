@triton.jit
def triton_poi_fused_add_0(in_ptr0, out_ptr0, znumel, ynumel, xnumel, ZBLOCK : tl.constexpr, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    znumel = 51
    ynumel = 51
    xnumel = 51
    zoffset = tl.program_id(2) * ZBLOCK
    zindex = zoffset + tl.arange(0, ZBLOCK)[None, None, :]
    zmask = zindex < znumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex
    z0 = zindex
    tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[51, 51, 51], strides=[1, 51, 2601], block_shape=[XBLOCK, YBLOCK, ZBLOCK], order=[2, 1, 0], offsets=[xoffset, yoffset, zoffset]), boundary_check=[0, 1, 2])
    tmp1 = tl.load(tl.make_block_ptr(in_ptr0, shape=[51, 51, 51], strides=[51, 1, 2601], block_shape=[XBLOCK, YBLOCK, ZBLOCK], order=[2, 1, 0], offsets=[xoffset, yoffset, zoffset]), boundary_check=[0, 1, 2])
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 + tmp0
    tmp4 = tmp2 + tmp3
    tl.store(tl.make_block_ptr(out_ptr0, shape=[51, 51, 51], strides=[1, 51, 2601], block_shape=[XBLOCK, YBLOCK, ZBLOCK], order=[2, 1, 0], offsets=[xoffset, yoffset, zoffset]), tl.broadcast_to(tmp4, [XBLOCK, YBLOCK, ZBLOCK]).to(tl.float32), boundary_check=[0, 1, 2])