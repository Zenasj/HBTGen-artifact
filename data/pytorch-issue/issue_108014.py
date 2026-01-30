import torch

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    arg1_1_size = arg1_1.size()
    s2 = arg1_1_size[3]
    s3 = arg1_1_size[4]
    assert_size_stride(arg0_1, (128, 128, 1, 5, 5), (3200, 25, 25, 5, 1))
    assert_size_stride(arg1_1, (8, 128, (-1) + s1, s2, s3), (((-128)*s2*s3) + (128*s1*s2*s3), ((-1)*s2*s3) + (s1*s2*s3), s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((128, 128, 1, 5, 5), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__self___conv_xy], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__to_copy_convolution_0.run(arg0_1, buf0, 409600, grid=grid(409600), stream=stream0)
        del arg0_1
        # Source Nodes: [l__self___conv_xy], Original ATen: [aten._to_copy, aten.convolution]
        buf1 = extern_kernels.convolution(arg1_1, buf0, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (8, 128, (-1) + s1, (-4) + s2, (-4) + s3), ((-2048) + (512*s2) + (512*s3) + (2048*s1) + ((-512)*s1*s2) + ((-512)*s1*s3) + ((-128)*s2*s3) + (128*s1*s2*s3), (-16) + (4*s2) + (4*s3) + (16*s1) + ((-1)*s2*s3) + ((-4)*s1*s2) + ((-4)*s1*s3) + (s1*s2*s3), 16 + ((-4)*s2) + ((-4)*s3) + (s2*s3), (-4) + s3, 1))
        del arg1_1
        return (buf1, )