import torch
import numpy as np
import time
import csv

batch_sizes = [1, 2, 4, 8]
matrix_sizes = [256, 512, 1024, 2048]
num_runs = 10
warmup_runs = 3

def run_int_mm(A, B):
    torch.mps.synchronize()
    start = time.perf_counter()
    c = A @ B
    torch.mps.synchronize()
    end = time.perf_counter()
    return c, end - start

results = {
    'N': [],
    'B': [],
    'mean_time': [],
    'std_time': []
}

for b in batch_sizes:
    for n in matrix_sizes:
        print(f"\nBenchmarking N={n} and B={b}")
        
        try:
            A_mps = torch.randint(low=-100, high=100, size=(b, n, n), dtype=torch.int8, device="mps")
            B_mps = torch.randint(low=-100, high=100, size=(b, n, n), dtype=torch.int8, device="mps")
            
            
            for _ in range(warmup_runs):
                _, _ = run_int_mm(A_mps, B_mps)
            
            times = []
            for _ in range(num_runs):
                _, t = run_int_mm(A_mps, B_mps)
                times.append(t)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            results['N'].append(n)
            results['B'].append(b)
            results['mean_time'].append(mean_time)
            results['std_time'].append(std_time)
            
            print(f"Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
            
        except RuntimeError as e:
            print(f"Error for N={n}: {e}")
            continue

with open('int_bmm_benchmark_times_new.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['N', 'batch', 'mean_time', 'std_time'])
    for i in range(len(results['N'])):
        writer.writerow([
            results['N'][i],
            results['B'][i],
            results['mean_time'][i],
            results['std_time'][i]
        ])