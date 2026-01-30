import torch

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, ), (1, ))
    assert_size_stride(arg1_1, (128, 8, 16, 64), (8192, 1024, 64, 1))
    assert_size_stride(arg2_1, (128, 8, 16, 64), (8192, 1024, 64, 1))
    assert_size_stride(arg3_1, (1024, 8, 64), (1536, 64, 1))
    assert_size_stride(arg4_1, (1024, 8, 64), (1536, 64, 1))
    # Source Nodes: [], Original ATen: []
    torch.ops.mylib.reshape_and_cache.default(arg4_1, arg3_1, arg2_1, arg1_1, arg0_1)
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    return ()

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 1, 128, 8192), (1048576, 1048576, 8192, 1))
    assert_size_stride(arg1_1, (1024, ), (1, ))
    assert_size_stride(arg2_1, (1024, 8, 64), (1536, 64, 1))
    assert_size_stride(arg3_1, (1024, 8, 64), (1536, 64, 1))
    buf0 = empty_strided_cpu((1048576, ), (1, ), torch.bfloat16)
    buf1 = empty_strided_cpu((2097152, ), (1, ), torch.bfloat16)
    cpp_fused_0(arg0_1, buf0, buf1)
    # Source Nodes: [], Original ATen: []
    torch.ops.mylib.reshape_and_cache.default(arg3_1, arg2_1, reinterpret_tensor(buf0, (128, 8, 16, 64), (8192, 1024, 64, 1), 0), reinterpret_tensor(buf1, (128, 8, 16, 64), (8192, 1024, 64, 1), 1048576), arg1_1)
    del arg1_1
    del arg2_1
    del arg3_1
    cpp_fused_1(buf1, buf0, arg0_1, arg0_1)
    del arg0_1
    return ()