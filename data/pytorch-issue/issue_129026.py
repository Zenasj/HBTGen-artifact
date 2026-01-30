py
@hl.generator(name="kernel")
class Kernel:
    in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
    out_ptr3 = hl.OutputBuffer(hl.Float(32), 2)

    def generate(g):
        in_ptr0 = g.in_ptr0
        out_ptr3 = g.out_ptr3
        xindex = hl.Var('xindex')
        rindex = hl.Var('rindex')
        r1 = rindex
        x0 = xindex
        idom = hl.RDom([hl.Range(0, 16), hl.Range(0, 32)])
        odom = hl.RDom([hl.Range(0, 16)])
        rdom = hl.RDom([hl.Range(0, 32)])
        xindex_idom = idom.x
        xindex_odom = odom.x
        rindex_idom = idom.y
        r1_idom = rindex_idom
        x0_idom = xindex_idom
        x0_odom = xindex_odom
        tmp0 = hl.Func('tmp0')
        tmp0[rindex, xindex] = in_ptr0[r1 + (32*x0)]
        tmp1 = hl.Func('tmp1')
        tmp1[xindex] = hl.maximum(rdom, tmp0[rdom, xindex])
        tmp2 = hl.Func('tmp2')
        tmp2[rindex, xindex] = tmp0[rindex, xindex] - tmp1[xindex]
        tmp3 = hl.Func('tmp3')
        tmp3[rindex, xindex] = hl.fast_exp(hl.cast(hl.Float(32), tmp2[rindex, xindex])) if tmp2.type().bits() <= 32 else hl.exp(tmp2[rindex, xindex])
        tmp4 = hl.Func('tmp4')
        tmp4[xindex] = hl.sum(rdom, tmp3[rdom, xindex])
        tmp5 = hl.Func('tmp5')
        tmp5[rindex, xindex] = tmp3[rindex, xindex] / tmp4[xindex]
        out_ptr3_i0 = hl.Var('out_ptr3_i0')
        out_ptr3_i1 = hl.Var('out_ptr3_i1')
        out_ptr3[out_ptr3_i0, out_ptr3_i1] = hl.cast(out_ptr3.type(), tmp5[out_ptr3_i0, out_ptr3_i1])

        assert g.using_autoscheduler()
        in_ptr0.set_estimates([hl.Range(0, 512)])
        out_ptr3.set_estimates([hl.Range(0, 32), hl.Range(0, 16)])

py
@hl.generator(name="kernel")
class Kernel:
    in_ptr0 = hl.InputBuffer(hl.Float(32), 2)
    out_ptr3 = hl.OutputBuffer(hl.Float(32), 2)

    def generate(g):
        in_ptr0 = g.in_ptr0
        out_ptr3 = g.out_ptr3
        h0 = hl.Var('h0')
        h1 = hl.Var('h1')
        rdom = hl.RDom([hl.Range(0, 32)])
        hr1 = rdom[0]
        tmp0 = hl.Func('tmp0')
        tmp0[h0, h1] = in_ptr0[h0, h1,]
        tmp1 = hl.Func('tmp1')
        tmp1[h1] = hl.maximum(rdom, tmp0[hr1, h1])
        tmp2 = hl.Func('tmp2')
        tmp2[h0, h1] = tmp0[h0, h1] - tmp1[h1]
        tmp3 = hl.Func('tmp3')
        tmp3[h0, h1] = hl.fast_exp(hl.cast(hl.Float(32), tmp2[h0, h1])) if tmp2.type().bits() <= 32 else hl.exp(tmp2[h0, h1])
        tmp4 = hl.Func('tmp4')
        tmp4[h1] = hl.sum(rdom, tmp3[hr1, h1])
        tmp5 = hl.Func('tmp5')
        tmp5[h0, h1] = tmp3[h0, h1] / tmp4[h1]
        out_ptr3[h0, h1,] = hl.cast(hl.Float(32), tmp5[h0, h1])

        assert g.using_autoscheduler()
        in_ptr0.dim(0).set_min(0)
        in_ptr0.dim(0).set_stride(1)
        in_ptr0.dim(0).set_extent(32)
        in_ptr0.dim(1).set_min(0)
        in_ptr0.dim(1).set_stride(32)
        in_ptr0.dim(1).set_extent(16)
        in_ptr0.set_estimates([hl.Range(0, 32), hl.Range(0, 16)])
        out_ptr3.set_estimates([hl.Range(0, 32), hl.Range(0, 16)])