import torch

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (20, ), (1, ))
    buf0 = empty((), device='cpu', dtype=torch.int64)
    cpp_fused_full_0(buf0)
    u0 = buf0.item()
    buf1 = None
    buf2 = buf0; del buf0  # reuse
    cpp_fused_full_sum_1(buf2, u0)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [add], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(arg0_1, buf2, buf3, 20, grid=grid(20), stream=stream0)
        run_intermediate_hooks('add', buf3)
        del arg0_1
        return (buf3, )

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (20, ), (1, ))
    eturn (arg0_1, )

r = torch.full((i0,), 1)

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (20, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((20, ), (1, ), torch.float16)
        # Source Nodes: [add, sum_1], Original ATen: [aten.add, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sum_0.run(arg0_1, buf2, 20, grid=grid(20), stream=stream0)
        run_intermediate_hooks('add', buf2)
        del arg0_1
        return (buf2, )