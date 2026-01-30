import torch
def bench_addcdiv(size=(32*1024**2, 5), device="cuda"):
    x=torch.rand(size, device=device, dtype=torch.float)
    y=torch.rand(size, device=device, dtype=torch.double)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
      torch.addcdiv(x, x, x, out=y)
    rc=prof.key_averages()
    print(rc)


if __name__ == "__main__":
    bench_addcdiv()