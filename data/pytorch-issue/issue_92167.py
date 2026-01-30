import torch

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(arg5_1, arg5_1, out=buf0)
        buf1 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(arg5_1, arg5_1, out=buf1)
        buf4 = empty_strided((8, 4), (4, 1), device='cuda', dtype=torch.float32)
        buf2 = as_strided(buf4, (4, 4), (4, 1))  # alias
        stream0 = get_cuda_stream(0)
        triton_fused_cat_0.run(buf0, buf2, 16, grid=grid(16), stream=stream0)
        del buf0
        buf3 = as_strided(buf4, (4, 4), (4, 1), 16)  # alias
        triton_fused_cat_0.run(buf1, buf3, 16, grid=grid(16), stream=stream0)
        del buf2
        del buf3
        buf5 = dist.all_reduce(buf4, async_op=True, group=0, op=ReduceOp.SUM)
        buf5.wait()
        del buf5
        buf7 = buf1; del buf1  # reuse
        extern_kernels.mm(arg5_1, arg5_1, out=buf7)
        del arg5_1
        buf8 = empty_strided((8, 4), (4, 1), device='cuda', dtype=torch.float32)
        triton_fused_add_1.run(buf4, buf7, buf8, 32, grid=grid(32), stream=stream0)
        return (buf8, )