import torch

dtype = torch.float32
dim = 1305301
ind_range = 100
a = torch.zeros(ind_range, device="cuda", dtype=dtype)
index = torch.randint(0, ind_range, (dim,), device="cuda")
src = torch.ones(dim, device="cuda", dtype=dtype)
print("=" * 20)
print(triton.testing.do_bench(
    lambda: a.scatter_add(0, index, src),
    return_mode="median",))
print("=" * 20)