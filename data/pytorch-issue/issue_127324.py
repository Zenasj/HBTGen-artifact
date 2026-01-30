import torch

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (3, ), (1, ))
    assert_size_stride(arg1_1, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [tensor], Original ATen: [_c10d_functional.all_reduce]
        stream0 = get_raw_stream(0)
        triton_poi_fused_all_reduce_0.run(arg0_1, buf0, 3, grid=grid(3), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [tensor], Original ATen: [_c10d_functional.all_reduce]
        torch.ops._c10d_functional.all_reduce_.default(buf0, 'sum', '0')
        buf5 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out2], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg1_1, buf5, 4, grid=grid(4), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [out1], Original ATen: [_c10d_functional.wait_tensor]
        torch.ops._c10d_functional.wait_tensor.default(buf0)
    return (buf0, buf5, )