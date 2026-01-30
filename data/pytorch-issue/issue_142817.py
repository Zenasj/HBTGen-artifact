import torch.nn as nn

import functools
import torch
import torch.nn.attention.flex_attention
import time
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_device("cuda")
torch.set_float32_matmul_precision("medium")


@torch.compile
def attn(query, key, value, global_causal_mask):
    n_batch, n_ctx, d_model = query.shape
    n_head = 16
    query = query.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    key = key.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    value = value.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    attn = functools.partial(
        torch.nn.attention.flex_attention.flex_attention,
        block_mask=global_causal_mask,
        return_lse=True,
    )
    x, _ = attn(query, key, value)
    x = x.transpose(1, 2).contiguous().reshape(n_batch, n_ctx, d_model)
    return x


def measure_time(seq_len):
    n_local_band = 128
    query = torch.randn(1, seq_len, 512, requires_grad=True)
    key = torch.randn(1, seq_len, 512, requires_grad=True)
    value = torch.randn(1, seq_len, 512, requires_grad=True)

    def global_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx > n_local_band)

    global_causal_mask = torch.nn.attention.flex_attention.create_block_mask(
        global_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
    )

    out = attn(query, key, value, global_causal_mask)
    out.sum().backward()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    out = attn(query, key, value, global_causal_mask)
    torch.cuda.synchronize()
    forward_time = time.perf_counter() - start_time

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    out.sum().backward()
    torch.cuda.synchronize()
    backward_time = time.perf_counter() - start_time

    return forward_time * 1000, backward_time * 1000  # Convert to milliseconds

seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
n_runs = 3

forward_times = []
backward_times = []

for seq_len in seq_lengths:
    print(f"Testing sequence length: {seq_len}")
    fwd_times = []
    bwd_times = []

    for _ in range(n_runs):
        f_time, b_time = measure_time(seq_len)
        fwd_times.append(f_time)
        bwd_times.append(b_time)

    forward_times.append(np.mean(fwd_times))
    backward_times.append(np.mean(bwd_times))

plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, forward_times, 'b-o', label='Forward Pass')
plt.plot(seq_lengths, backward_times, 'r-o', label='Backward Pass')

plt.xlabel('Sequence Length')
plt.ylabel('Time (ms)')
plt.title('Attention Performance Scaling with Sequence Length')
plt.grid(True)
plt.legend()

for i, seq_len in enumerate(seq_lengths):
    plt.annotate(f'{forward_times[i]:.1f}ms',
                (seq_len, forward_times[i]),
                textcoords="offset points",
                xytext=(0,10),
                ha='center')
    plt.annotate(f'{backward_times[i]:.1f}ms',
                (seq_len, backward_times[i]),
                textcoords="offset points",
                xytext=(0,-15),
                ha='center')

plt.tight_layout()
plt.savefig('attention_scaling.png', dpi=300, bbox_inches='tight')
plt.show()

import functools
import torch
import torch.nn.attention.flex_attention
import time
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

torch.set_default_device("cuda")
torch.set_float32_matmul_precision("medium")

torch._dynamo.config.cache_size_limit = 1000



@torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
def attn(query, key, value, global_causal_mask):
    n_batch, n_ctx, d_model = query.shape
    n_head = 16
    query = query.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    key = key.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    value = value.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    attn = functools.partial(
        torch.nn.attention.flex_attention.flex_attention,
        block_mask=global_causal_mask,
        return_lse=True,
    )
    x, _ = attn(query, key, value)
    x = x.transpose(1, 2).contiguous().reshape(n_batch, n_ctx, d_model)
    return x


def measure_time(seq_len):
    n_local_band = 128
    query = torch.randn(1, seq_len, 512, requires_grad=True)
    key = torch.randn(1, seq_len, 512, requires_grad=True)
    value = torch.randn(1, seq_len, 512, requires_grad=True)

    def global_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx > n_local_band)

    global_causal_mask = torch.nn.attention.flex_attention.create_block_mask(
        global_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
    )

    out = attn(query, key, value, global_causal_mask)
    out.sum().backward()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    out = attn(query, key, value, global_causal_mask)
    torch.cuda.synchronize()
    forward_time = time.perf_counter() - start_time

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    out.sum().backward()
    torch.cuda.synchronize()
    backward_time = time.perf_counter() - start_time

    return forward_time * 1000, backward_time * 1000  # Convert to milliseconds

seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
n_runs = 3

# Lists to store all timing results
results = []

for seq_len in seq_lengths:
    print(f"Testing sequence length: {seq_len}")
    fwd_times = []
    bwd_times = []

    for run in range(n_runs):
        f_time, b_time = measure_time(seq_len)
        fwd_times.append(f_time)
        bwd_times.append(b_time)
        
        # Store individual run results
        results.append([
            seq_len,
            run + 1,
            f"{f_time:.2f}",
            f"{b_time:.2f}",
            f"{(f_time + b_time):.2f}"
        ])

# Print detailed results table
headers = ["Sequence Length", "Run", "Forward (ms)", "Backward (ms)", "Total (ms)"]
print("\nDetailed Results:")
print(tabulate(results, headers=headers, tablefmt="grid"))

# Calculate and print averages
avg_results = []
for seq_len in seq_lengths:
    seq_results = [r for r in results if r[0] == seq_len]
    avg_fwd = np.mean([float(r[2]) for r in seq_results])
    avg_bwd = np.mean([float(r[3]) for r in seq_results])
    avg_results.append([
        seq_len,
        f"{avg_fwd:.2f}",
        f"{avg_bwd:.2f}",
        f"{(avg_fwd + avg_bwd):.2f}"
    ])

print("\nAveraged Results:")
headers = ["Sequence Length", "Avg Forward (ms)", "Avg Backward (ms)", "Avg Total (ms)"]
print(tabulate(avg_results, headers=headers, tablefmt="grid"))

# Plotting code remains the same
plt.figure(figsize=(10, 6))
forward_times = [float(r[1]) for r in avg_results]
backward_times = [float(r[2]) for r in avg_results]

plt.plot(seq_lengths, forward_times, 'b-o', label='Forward Pass')
plt.plot(seq_lengths, backward_times, 'r-o', label='Backward Pass')

plt.xlabel('Sequence Length')
plt.ylabel('Time (ms)')
plt.title('Attention Performance Scaling with Sequence Length')
plt.grid(True)
plt.legend()

for i, seq_len in enumerate(seq_lengths):
    plt.annotate(f'{forward_times[i]:.1f}ms',
                (seq_len, forward_times[i]),
                textcoords="offset points",
                xytext=(0,10),
                ha='center')
    plt.annotate(f'{backward_times[i]:.1f}ms',
                (seq_len, backward_times[i]),
                textcoords="offset points",
                xytext=(0,-15),
                ha='center')

plt.tight_layout()
plt.savefig('attention_scaling.png', dpi=300, bbox_inches='tight')
plt.show()