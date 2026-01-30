import torch
import numpy as np
import time
import csv


matrix_sizes = [3, 4, 32, 64, 128, 256, 512, 1024]
num_runs = 10
warmup_runs = 3

def create_invertible_matrix(n):
    """
    Create a random invertible matrix.
    To help ensure the matrix is well-conditioned, we add a scaled identity matrix.
    """
    torch.manual_seed(42)
    A = torch.randn(1, n, n, dtype=torch.float32)
    return A + n * torch.eye(n).expand(1, -1, -1)

def run_inv_mps(A):
    """
    Run torch.linalg.inv on the given matrix using the MPS device.
    """
    torch.mps.synchronize()
    start = time.perf_counter()
    B = torch.linalg.inv(A)
    torch.mps.synchronize()
    end = time.perf_counter()
    return B, end - start

results = {
    'N': [],
    'mean_time': [],
    'std_time': []
}

for n in matrix_sizes:
    A_cpu = create_invertible_matrix(n)
    A_mps = A_cpu.to("mps")
    
    for _ in range(warmup_runs):
        _, _ = run_inv_mps(A_mps)
    
    times = []
    for _ in range(num_runs):
        _, t = run_inv_mps(A_mps)
        times.append(t)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    results['N'].append(n)
    results['mean_time'].append(mean_time)
    results['std_time'].append(std_time)
    
    print(f"Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
    

with open('temp.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['N', 'mean_time', 'std_time'])
    for i in range(len(results['N'])):
        writer.writerow([
            results['N'][i],
            results['mean_time'][i],
            results['std_time'][i]
        ])