import torch

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 2048, 128), (262144, 262144, 128, 1))
    assert_size_stride(arg1_1, (1, 1, 2048, 128), (262144, 262144, 128, 1))
    assert_size_stride(arg2_1, (1, 1, 2048, 128), (262144, 262144, 128, 1))
    assert_size_stride(arg3_1, (1, 1, 16), (16, 16, 1))
    assert_size_stride(arg4_1, (1, 1, 16, 16), (256, 256, 16, 1))
    assert_size_stride(arg5_1, (1, 1, 16), (16, 16, 1))
    assert_size_stride(arg6_1, (1, 1, 16, 16), (256, 256, 16, 1))
    assert_size_stride(arg7_1, (1, 1, 16), (16, 16, 1))
    assert_size_stride(arg8_1, (1, 1, 16, 16), (256, 256, 16, 1))
    assert_size_stride(arg9_1, (1, 1, 16), (16, 16, 1))
    assert_size_stride(arg10_1, (1, 1, 16, 16), (256, 256, 16, 1))
    buf0 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.float32) # TODO: ERROR here. Should be cuda:1
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        buf1 = empty_strided_cuda((1, 1, 2048, 128), (262144, 262144, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [flex_attention], Original ATen: []
        stream1 = get_raw_stream(1)
        breakpoint()
        triton_tem_fused_0.run(arg0_1, arg1_1, arg2_1, buf0, arg3_1, arg4_1, arg5_1, arg6_1, buf1, grid=torch._inductor.kernel.flex_attention.flex_attention_grid(1, 1, 2048, 128, meta0), stream=stream1)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del buf0
    return (buf1, )

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 2048, 128), (262144, 262144, 128, 1))
    assert_size_stride(arg1_1, (1, 1, 2048, 128), (262144, 262144, 128, 1))
    assert_size_stride(arg2_1, (1, 1, 2048, 128), (262144, 262144, 128, 1))
    assert_size_stride(arg3_1, (1, 1, 16), (16, 16, 1))
    assert_size_stride(arg4_1, (1, 1, 16, 16), (256, 256, 16, 1))
    assert_size_stride(arg5_1, (1, 1, 16), (16, 16, 1))
    assert_size_stride(arg6_1, (1, 1, 16, 16), (256, 256, 16, 1))
    assert_size_stride(arg7_1, (1, 1, 16), (16, 16, 1))
    assert_size_stride(arg8_1, (1, 1, 16, 16), (256, 256, 16, 1))
    assert_size_stride(arg9_1, (1, 1, 16), (16, 16, 1))
    assert_size_stride(arg10_1, (1, 1, 16, 16), (256, 256, 16, 1))
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        buf0 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.float32) # New: move into device guard
        buf1 = empty_strided_cuda((1, 1, 2048, 128), (262144, 262144, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [flex_attention], Original ATen: []
        stream1 = get_raw_stream(1)
        triton_tem_fused_0.run(arg0_1, arg1_1, arg2_1, buf0, arg3_1, arg4_1, arg5_1, arg6_1, buf1, grid=torch._inductor.kernel.flex_attention.flex_attention_grid(1, 1, 2048, 128, meta0), stream=stream1)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del buf0
    return (buf1, )