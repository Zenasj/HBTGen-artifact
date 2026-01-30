def kernel45(
    x,
    w,
    # stride of tensor
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    stride_biasn,
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    # parameters of conv
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    output_padding_h: tl.constexpr,
    output_padding_w: tl.constexpr,
    groups: tl.constexpr,
    # pointer inc for x
    delta_x_ptr,
    # fusable kernels args
    in_ptr0,
    in_ptr1,
    in_ptr2,
    in_ptr3,
    in_ptr4,
    in_ptr5,
    in_ptr6,
    in_ptr7,
    in_ptr8,
    in_ptr9,
    in_ptr10,
    in_ptr11,
    in_ptr12,
    out_ptr3,
    out_ptr4,
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    CONV1X1_NHWC: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W
    off_y_w = off_y_hw % OUT_W

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W
    # load inc ptr of x, upade x_ptrs
    if not CONV1X1_NHWC:
        delta_x_ptrs = delta_x_ptr + off_x_crs
        off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS, other=0)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]

    mask_x = (
        (off_x_n < BATCH)
        & (off_x_h >= 0)
        & (off_x_h < IN_H)
        & (off_x_w >= 0)
        & (off_x_w < IN_W)
    )[:, None] & (off_x_crs < CRS)[None, :]

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):

        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w)
        # ------ update ptrs ------
        w_ptrs += BLOCK_K
        # load inc ptr of x, upade x_ptrs
        if not CONV1X1_NHWC:
            delta_x_ptrs += BLOCK_K
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS, other=0)
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            x_ptrs += BLOCK_K

        mask_x = (
            (off_x_n < BATCH)
            & (off_x_h >= 0)
            & (off_x_h < IN_H)
            & (off_x_w >= 0)
            & (off_x_w < IN_W)
        )[:, None] & (off_x_crs < CRS)[None, :]
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = acc.to(out_ptr3.dtype.element_ty)


    XBLOCK: tl.constexpr = BLOCK_M
    YBLOCK: tl.constexpr = BLOCK_N
    xnumel = BATCH * (OUT_H + 2 * output_padding_h) * (OUT_W + 2 * output_padding_w)
    ynumel = KERNEL_N
    xoffset = pid_nhw * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    yoffset = pid_k * YBLOCK
    yindex = yoffset + tl.reshape(tl.arange(0, YBLOCK), [1, YBLOCK])
    ymask = yindex < ynumel
    x0 = xindex
    y1 = yindex
    tmp0 = tl.load(in_ptr0 + y1 + (256*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp19 = tl.load(in_ptr5 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp25 = tl.load(in_ptr6 + y1 + (256*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp26 = tl.load(in_ptr7 + y1 + (256*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp27 = tl.load(in_ptr8 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp30 = tl.load(in_ptr9 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp33 = tl.load(in_ptr10 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp40 = tl.load(in_ptr11 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp43 = tl.load(in_ptr12 + y1 + tl.zeros([XBLOCK, YBLOCK], tl.int32), xmask & ymask).to(tl.float32)
    tmp2 = acc + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 - tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp0 + tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 - tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 + tmp9
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = tmp37 * tmp13
    tmp39 = tmp32 * tmp38
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 * tmp41
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 + tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp25 + tmp46
    tl.store(out_ptr3 + y1 + (256*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32), tmp24, xmask & ymask)
    tl.store(out_ptr4 + y1 + (256*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32), tmp47, xmask & ymask)