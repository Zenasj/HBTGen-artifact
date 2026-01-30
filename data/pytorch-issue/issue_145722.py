import torch
import numpy as np
import time
import csv

matrix_sizes = [512, 1024, 2048, 4096]
batch_sizes = [1, 2, 4, 8, 16]
num_runs = 10
warmup_runs = 3

def create_spd_matrix(n, batch_size):
    torch.manual_seed(42)
    A = torch.randn(batch_size, n, n, dtype=torch.float32)
    return A @ A.transpose(-2, -1) + n * torch.eye(n).expand(batch_size, -1, -1)

def run_cholesky_mps(A):
    torch.mps.synchronize()
    start = time.perf_counter()
    b = torch.linalg.cholesky(A, upper=False)
    torch.mps.synchronize()
    end = time.perf_counter()
    return b, end - start

results = {
    'N': [],
    'batch_size': [],
    'mean_time': [],
    'std_time': []
}

for n in matrix_sizes:
    for batch_size in batch_sizes:
        print(f"\nBenchmarking N={n}, batch_size={batch_size}")
        
        try:
            A_cpu = create_spd_matrix(n, batch_size)
            A_mps = A_cpu.to("mps")
            
            for _ in range(warmup_runs):
                _, _ = run_cholesky_mps(A_mps)
            
            times = []
            for _ in range(num_runs):
                _, t = run_cholesky_mps(A_mps)
                times.append(t)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            results['N'].append(n)
            results['batch_size'].append(batch_size)
            results['mean_time'].append(mean_time)
            results['std_time'].append(std_time)
            
            print(f"Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
            
        except RuntimeError as e:
            print(f"Error for N={n}, batch_size={batch_size}: {e}")
            continue

with open('cholesky_benchmark_times.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['N', 'batch_size', 'mean_time', 'std_time'])
    for i in range(len(results['N'])):
        writer.writerow([
            results['N'][i],
            results['batch_size'][i],
            results['mean_time'][i],
            results['std_time'][i]
        ])