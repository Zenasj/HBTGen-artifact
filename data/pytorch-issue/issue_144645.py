import torch

DEVICE=torch.device('cuda')

# Time cost for near 1024
for cnt in range(1020, 1030):
    x = torch.randn(4096, cnt, device=DEVICE, dtype=torch.float32)
    #x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    #warm up
    need_warmup = True
    round = 5
    if need_warmup:
        for _ in range(round):
            output = torch.softmax(x, dim=-1)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    # Start time
    start_time.record()

    # Apply softmax
    for _ in range(round):
        output = torch.softmax(x, dim=-1)

    # End time
    end_time.record()

    torch.cuda.synchronize()

    # Calculate elapsed time
    elapsed_time_ms = start_time.elapsed_time(end_time)
    # print(f"CUDA Time: {elapsed_time_ms:.6f} ms")
    gbps = lambda ms: round * 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    print(f"n as {cnt} of softmax: {gbps(elapsed_time_ms):.6f} gb/s")