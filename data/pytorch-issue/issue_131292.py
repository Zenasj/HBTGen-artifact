def masked_mean(
    input: torch.Tensor,
    mask: torch.Tensor,
    dim: int | torch.types._size | None = None,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return (input * mask).sum(dim=dim, keepdim=keepdim, dtype=dtype or input.dtype) / mask.broadcast_to(input.shape).sum(dim=dim, keepdim=keepdim)

count = inmask.sum(dim=dim, keepdim=keepdim)

import time

import matplotlib.pyplot as plt
import torch
import torch.types

num_elements: list[int] = []
avg_times_custom: list[float] = []
avg_times_torch_masked_mean: list[float] = []

# takes about 3 minutes on my laptop
NUM_STEPS = 16
RUNS_PER_STEP = 5
MAX_NUM_ELEMENTS = 4096 * 4096 * 32


def masked_mean(
    input: torch.Tensor,
    mask: torch.Tensor,
    dim: int | torch.types._size | None = None,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return (input * mask).sum(dim=dim, keepdim=keepdim, dtype=dtype or input.dtype) / mask.broadcast_to(input.shape).sum(dim=dim, keepdim=keepdim)


for n_elements in torch.linspace(1, MAX_NUM_ELEMENTS, NUM_STEPS):
    n_elements = int(n_elements)
    num_elements.append(n_elements)

    input_tensor = torch.randn(n_elements, dtype=torch.float32)
    mask = torch.tensor(True)
    # mask = torch.randint_like(input_tensor, 0, 2, dtype=torch.bool)  # non-trivial mask

    torch.testing.assert_close(masked_mean(input_tensor, mask=mask), torch.masked.mean(input_tensor, mask=mask))  # will fail if not mask.any()

    total_time_custom = 0.0

    for _ in range(RUNS_PER_STEP):
        start_time = time.perf_counter()
        masked_mean(input_tensor, mask=mask)
        end_time = time.perf_counter()
        total_time_custom += (end_time - start_time)

    avg_time_cusotm = (total_time_custom / RUNS_PER_STEP)
    avg_times_custom.append(avg_time_cusotm)

    total_time_torch_masked_mean = 0.0

    for _ in range(RUNS_PER_STEP):
        start_time = time.perf_counter()
        torch.masked.mean(input_tensor, mask=mask)
        end_time = time.perf_counter()
        total_time_torch_masked_mean += (end_time - start_time)

    avg_time_torch_masked_mean = (total_time_torch_masked_mean / RUNS_PER_STEP)
    avg_times_torch_masked_mean.append(avg_time_torch_masked_mean)

plt.figure(figsize=(10, 6))
plt.plot(num_elements, avg_times_custom, label='proposed implementation (without new_ones)')
plt.plot(num_elements, avg_times_torch_masked_mean, label='torch.masked.mean')
plt.scatter(num_elements, avg_times_custom)
plt.scatter(num_elements, avg_times_torch_masked_mean)
plt.xlabel('Number of Elements')
plt.ylabel('Average Execution Time (s)')
plt.title('Average Execution Time Scaling')
plt.legend()
plt.show()

import time

import matplotlib.pyplot as plt
import torch
import torch.types

num_elements: list[int] = []
avg_times_custom: list[float] = []
avg_times_torch_masked_mean: list[float] = []

# takes about 1.5 minutes on my laptop
NUM_STEPS = 16
RUNS_PER_STEP = 5
MAX_NUM_ELEMENTS = 4096 * 4096 * 32

def masked_mean(
    input: torch.Tensor,
    dim: int | torch.types._size | None = None,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return input.mean(dim=dim, keepdim=keepdim, dtype=dtype)


for n_elements in torch.linspace(1, MAX_NUM_ELEMENTS, NUM_STEPS):
    n_elements = int(n_elements)
    num_elements.append(n_elements)

    input_tensor = torch.randn(n_elements, dtype=torch.float32)

    torch.testing.assert_close(masked_mean(input_tensor), torch.masked.mean(input_tensor, mask=None))

    total_time_custom = 0.0

    for _ in range(RUNS_PER_STEP):
        start_time = time.perf_counter()
        masked_mean(input_tensor)
        end_time = time.perf_counter()
        total_time_custom += (end_time - start_time)

    avg_time_cusotm = (total_time_custom / RUNS_PER_STEP)
    avg_times_custom.append(avg_time_cusotm)

    total_time_torch_masked_mean = 0.0

    for _ in range(RUNS_PER_STEP):
        start_time = time.perf_counter()
        torch.masked.mean(input_tensor, mask=None)
        end_time = time.perf_counter()
        total_time_torch_masked_mean += (end_time - start_time)

    avg_time_torch_masked_mean = (total_time_torch_masked_mean / RUNS_PER_STEP)
    avg_times_torch_masked_mean.append(avg_time_torch_masked_mean)

plt.figure(figsize=(10, 6))
plt.plot(num_elements, avg_times_custom, label='proposed implementation (torch.mean)')
plt.plot(num_elements, avg_times_torch_masked_mean, label='torch.masked.mean(mask=None)')
plt.scatter(num_elements, avg_times_custom)
plt.scatter(num_elements, avg_times_torch_masked_mean)
plt.xlabel('Number of Elements')
plt.ylabel('Average Execution Time (s)')
plt.title('Average Execution Time Scaling')
plt.legend()
plt.show()