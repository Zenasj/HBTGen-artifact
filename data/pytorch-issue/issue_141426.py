(python)
import torch

t = torch.zeros((5, 3, 6), dtype=torch.complex128, device='cuda')

@torch.compile
def func(a):
    print("a:")
    print("size:", a.shape)
    print("stride:", a.stride())
    b = torch.ops.aten._fft_c2c.default(a, [1], 1, True)
    print("b:")
    print("size:", b.shape)
    print("stride:", b.stride())
    c = torch.ops.aten._conj_physical.default(b)
    print("c:")
    print("size:", c.shape)
    print("stride:", c.stride())
    return c

func(t)

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (5, 3, 6), (18, 1, 3))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [buf1], Original ATen: [aten._conj_physical]
        buf0 = torch.ops.aten._conj_physical.default(arg0_1)
        del arg0_1
        buf1 = buf0
        del buf0
    return (buf1, )

def call(args):
    tangents_1, = args
    args.clear()
    assert_size_stride(tangents_1, (5, 3, 6), (18, 6, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf0 = torch.ops.aten._fft_c2c.default(tangents_1, [1], 1, True)
        assert_size_stride(buf0, (5, 3, 6), (18, 1, 3))
        del tangents_1
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj_physical]
        buf1 = torch.ops.aten._conj_physical.default(buf0)
        assert_size_stride(buf1, (5, 3, 6), (18, 6, 1))
        del buf0
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.new_zeros]
        buf2 = torch.ops.aten.full.default([5, 3, 10], 0, dtype=torch.complex128, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        assert_size_stride(buf2, (5, 3, 10), (30, 10, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice]
        buf3 = torch.ops.aten.slice.Tensor(buf2, 2, 0, 6)
        assert_size_stride(buf3, (5, 3, 6), (30, 10, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.copy]
        buf4 = torch.ops.aten.copy.default(buf3, buf1)
        assert_size_stride(buf4, (5, 3, 6), (30, 10, 1))
        del buf1
        del buf3
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf5 = torch.ops.aten.slice_scatter.default(buf2, buf4, 2, 0, 6)
        assert_size_stride(buf5, (5, 3, 10), (30, 10, 1))
        del buf2
        del buf4
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf6 = torch.ops.aten._fft_c2c.default(buf5, [2], 1, False)
        assert_size_stride(buf6, (5, 3, 10), (30, 10, 1))
        del buf5
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_real]
        buf7 = torch.ops.aten.view_as_real.default(buf6)
        assert_size_stride(buf7, (5, 3, 10, 2), (60, 20, 2, 1))
        buf8 = empty_strided_cuda((5, 6, 7), (42, 7, 1), torch.float64)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_slice_backward_0.run(buf7, buf8, 210, grid=grid(210), stream=stream0)
        del buf6
        del buf7
    return (buf8, )