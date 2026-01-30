import torch

def fn(t1, t2, t3, t4, i1: int, i2: int):
    v1 = torch.sub(t3, t4, alpha=i2)
    v2 = torch.mul(v1, t2)
    v3 = torch.reshape(t1, [1, 12, 64, 4096])
    v4 = torch.add(v3, v2, alpha=i1)
    v5 = torch._softmax(v4, -1, False)
    v6 = torch.reshape(v5, [12, 64, 4096])
    return v6

with torch.jit.fuser("fuser2"):
    fn_s = torch.jit.script(fn)
    other_dtype = torch.double
    t1 = torch.rand((12, 64, 4096), dtype=torch.float).cuda()
    t2 = torch.rand((), dtype=other_dtype).cuda()
    t3 = torch.rand((), dtype=other_dtype).cuda()
    t4 = torch.rand((1, 1, 1, 4096), dtype=torch.float).cuda()

    fn_s(t1, t2, t3, t4, 1, 1)
    fn_s(t1, t2, t3, t4, 1, 1)