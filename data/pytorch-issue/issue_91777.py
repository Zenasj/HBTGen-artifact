import torch

torch.set_float32_matmul_precision('high')

def gemm(x, y):
    return x @ y

gemm_opt = torch.compile(gemm)

x2 = torch.randn(1024, 1024, device="cuda:2")
y2 = torch.randn(1024, 1024, device="cuda:2")

print(gemm(x2, y2).device)
print(gemm_opt(x2, y2).device)

x3 = x2.to("cuda:3")
y3 = y2.to("cuda:3")
print(gemm(x3, y3).device)
print(gemm_opt(x3, y3).device)

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    with torch.cuda.device(2):
        buf0 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        aten.mm.out(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        return (buf0, )