import torch.nn as nn

def print_stats(y, y_p):
    abs_diff = (y - y_p).abs()
    items_different = (abs_diff > 1e-3).int().sum()

    print(
        f"linear diff sum: {abs_diff.sum()}"
    )
    print(
        f"items different (> 1e-3): {items_different} ({100*(int(items_different) / y.numel())}%)"
        )


x_cpu = torch.rand((253, 144, 768), dtype=torch.float32)

# This works, it seems like if you go higher than 129 in
# the second dimension, the errors start piling up.
# x_cpu = torch.rand((253, 129, 768), dtype=torch.float32)

linear_cpu = nn.Linear(in_features=x_cpu.shape[-1], out_features=3 * x_cpu.shape[-1])
linear_cpu.eval()
y_cpu = linear_cpu(x_cpu)

x_mps = x_cpu.to("mps")
linear_mps = linear_cpu.to("mps")

print("=== MPS Linear ===")
y_mps = linear_mps(x_mps)
print_stats(y_mps.to("cpu"), y_cpu)

print("=== MPS matmul ===")
y_mps_matmul = x_mps @ linear_mps.weight.t() + linear_mps.bias
print_stats(y_mps_matmul.to("cpu"), y_cpu)