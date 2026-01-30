import torch
import torch.utils.benchmark as benchmark

# Prepare matrices
N = 10000
A = torch.randn(N, N)
B = torch.randn(N, N)

# Send to device
device = torch.device("cuda")
A = A.to(device)
B = B.to(device)

# Benchmark
n_threads = 1
t = benchmark.Timer(
    stmt="A @ B",
    globals={"A": A, "B": B},
    num_threads=n_threads,
)
m = t.blocked_autorange(min_run_time=3)

# Print results
print(m)
print(f"Mean:  {m.mean * 1e3:6.2f} ms"
      + f" | First: {m._sorted_times[0] *1e3:6.2f} ms"
      + f" | Median: {m.median *1e3:6.2f} ms"
      + f" | Last: {m._sorted_times[-1] *1e3:6.2f} ms.")