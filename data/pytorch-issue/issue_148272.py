import torch
import numpy as np
import time
import csv

matrix_sizes = [1, 100, 1000, 10_000]
num_runs = 1000
warmup_runs = 3

def run_sqrt(A):
    torch.mps.synchronize()
    start = time.perf_counter()
    c = torch.sqrt(A)
    torch.mps.synchronize()
    end = time.perf_counter()
    return c, end - start

results = {
    'N': [],
    'mean_time': [],
    'std_time': []
}

for n in matrix_sizes:
    print(f"\nBenchmarking N={n}")
    
    try:
        A_mps = torch.rand((n, n), dtype=torch.float32, device="mps")
        
        for _ in range(warmup_runs):
            _, _ = run_sqrt(A_mps)
        
        times = []
        for _ in range(num_runs):
            _, t = run_sqrt(A_mps)
            times.append(t)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        results['N'].append(n)
        results['mean_time'].append(mean_time)
        results['std_time'].append(std_time)
        
        print(f"Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
        
    except RuntimeError as e:
        print(f"Error for N={n}: {e}")
        continue

with open('sqrt_benchmark_times_new.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['N', 'mean_time', 'std_time'])
    for i in range(len(results['N'])):
        writer.writerow([
            results['N'][i],
            results['mean_time'][i],
            results['std_time'][i]
        ])