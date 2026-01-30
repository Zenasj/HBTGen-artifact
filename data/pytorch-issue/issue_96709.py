def triton_(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks0) % ks0
    x0 = xindex % ks0
    x2 = (xindex // ks2)
    x4 = xindex
    tmp0 = (-1) + x1 + ((-1)*((((-1)*((((-3) + ks1) // 2))) // 2))) + ((-1)*(tl.libdevice.ceil(((1/2) + ((1/2)*((((-3) + ks1) // 2)))))))
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 165
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0 + ((-1)*((((-1)*((((-3) + ks1) // 2))) // 2))) + ((-1)*(tl.libdevice.ceil(((1/2) + ((1/2)*((((-3) + ks1) // 2)))))))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-2) + x0 + x1 + x2 + ((-1)*((((-3) + ks1) // 2))) + ((-2)*((((-1)*((((-3) + ks1) // 2))) // 2))) + ((-2)*(tl.libdevice.ceil(((1/2) + ((1/2)*((((-3) + ks1) // 2))))))) + (x1*((((-3) + ks1) // 2))) + (x2*(((((-3) + ks1) // 2))*((((-3) + ks1) // 2)))) + ((-1)*((((-1)*((((-3) + ks1) // 2))) // 2))*((((-3) + ks1) // 2))) + ((-1)*((((-3) + ks1) // 2))*(tl.libdevice.ceil(((1/2) + ((1/2)*((((-3) + ks1) // 2))))))) + (2*x2*((((-3) + ks1) // 2))) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0)