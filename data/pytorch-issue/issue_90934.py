import torch

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    with torch.cuda.device(1):
        buf0 = empty_strided((4, ), (1, ), device='cuda', dtype=torch.float32)
        stream1 = get_cuda_stream(1)
        triton_fused_div_0.run(arg0_1, arg1_1, buf0, 4, grid=grid(4), stream=stream1)
        del arg0_1
        del arg1_1
        return (buf0, )