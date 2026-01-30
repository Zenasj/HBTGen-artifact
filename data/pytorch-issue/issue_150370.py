import torch

def get_profiler():
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
    )

def profile_tensor_ops():
    device = "cuda"
    with get_profiler() as prof:
        for _ in range(5):
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


if __name__ == "__main__":
    profile_tensor_ops()