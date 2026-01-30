import torch.nn as nn

import csv
import time

import numpy as np

import torch
import torch.nn.functional as F


matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
batch_sizes = [1]
elementwise_affine = [False, True]
num_runs = 50
warmup_runs = 3


def create_input_tensor(n, batch_size):
    torch.manual_seed(42)
    return torch.randn(batch_size, n, dtype=torch.float32)


def run_layer_norm(A, normalized_shape, elementwise_affine):
    torch.mps.synchronize()
    start = time.perf_counter()
    out = F.layer_norm(A, normalized_shape)
    torch.mps.synchronize()
    end = time.perf_counter()
    return out, end - start


results = {"N": [], "elementwise_affine": [], "batch_size": [], "mean_time": [], "std_time": []}

for el_aff in elementwise_affine:
    for n in matrix_sizes:
        for batch_size in batch_sizes:
            print(f"\nBenchmarking LayerNorm for input size N={n}, batch_size={batch_size}, elementwise_affine={el_aff}")

            try:
                A_cpu = create_input_tensor(n, batch_size)
                A_mps = A_cpu.to("mps")

                normalized_shape = (n,)

                for _ in range(warmup_runs):
                    _, _ = run_layer_norm(A_mps, normalized_shape, el_aff)

                times = []
                for _ in range(num_runs):
                    _, t = run_layer_norm(A_mps, normalized_shape, el_aff)
                    times.append(t)

                mean_time = np.mean(times)
                std_time = np.std(times)

                results["N"].append(n)
                results["elementwise_affine"].append(el_aff)
                results["batch_size"].append(batch_size)
                results["mean_time"].append(mean_time)
                results["std_time"].append(std_time)

                print(f"Mean time: {mean_time:.4f}s ± {std_time:.4f}s")

            except RuntimeError as e:
                print(f"Error for N={n}, batch_size={batch_size}: {e}")
                continue

with open("layernorm_benchmark_times_new.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "elementwise_affine", "batch_size", "mean_time", "std_time"])
    for i in range(len(results["N"])):
        writer.writerow(
            [
                results["N"][i],
                results["elementwise_affine"][i],
                results["batch_size"][i],
                results["mean_time"][i],
                results["std_time"][i],
            ]
        )