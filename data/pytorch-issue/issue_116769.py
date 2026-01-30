import torch

def test_matmul_transpose_associativity(device):
    # Create a random matrix 16384x128
    x = torch.rand(16384, 128, device=device)
    # Create a random matrix 70000x128
    y = torch.rand(128, 70000,device=device)

    # Sample 23444 random indices from 70000
    indices = torch.randperm(70000, device=device)[:23444]

    # Select 23444 random rows from y
    y_sampled = y[:, indices]

    # Multiply x by y_sampled
    z_sampled = torch.matmul(x, y_sampled)
    # Multiply x by y
    z = torch.matmul(x, y)

    # z[:, indices] should be equal to z_sampled
    assert torch.allclose(z[:, indices], z_sampled, rtol=1e-2, atol=1e-2), "Matmul associativity test failed on device {}".format(device)

test_matmul_transpose_associativity("cpu")
test_matmul_transpose_associativity("mps")

import torch

def test_matmul_transpose_associativity(device):
    # Create a random matrix 16384x1
    x = torch.rand(16384, 1, device=device)
    # Create a random matrix 1x70000
    y = torch.rand(1, 70000,device=device)

    # Sample 23444 random indices from 70000
    indices = torch.randperm(70000, device=device)[:23444]

    # Select 23444 random rows from y
    y_sampled = y[:, indices]

    # Multiply x by y_sampled
    z_sampled = torch.matmul(x, y_sampled)
    # Multiply x by y
    z = torch.matmul(x, y)

    # z[:, indices] should be equal to z_sampled
    assert torch.allclose(z[:, indices], z_sampled, rtol=0.5, atol=0.5), "Matmul associativity test failed on device {}".format(device)

test_matmul_transpose_associativity("cpu")
test_matmul_transpose_associativity("mps")

import torch.nn as nn
import torch
def print_stats(y, y_p):
    abs_diff = (y - y_p).abs()
    items_different = (abs_diff > 1e-3).int().sum()

    print(
        f"linear diff sum: {abs_diff.sum()}"
    )
    print(
        f"items different (> 1e-3): {items_different} ({100*(int(items_different) / y.numel())}%)"
        )


x_cpu = torch.rand((1025, 1), dtype=torch.float32)
linear_cpu = nn.Linear(in_features=1, out_features=32769,bias=False)
linear_cpu.eval()
y_cpu = linear_cpu(x_cpu)
print(y_cpu.shape)
print("=== CPU matmul ===")
y_matmul = torch.matmul(x_cpu,linear_cpu.weight.t())
print_stats(y_matmul, y_cpu)

x_mps = x_cpu.to("mps")
linear_mps = linear_cpu.to("mps")

print("=== MPS Linear ===")
y_mps = linear_mps(x_mps)
print_stats(y_mps.to("cpu"), y_cpu)

print("=== MPS matmul ===")
y_mps_matmul = torch.matmul(x_mps,linear_mps.weight.t())
print_stats(y_mps_matmul.to("cpu"), y_cpu)

import torch

x=torch.rand(10, 1, device="mps")
y=torch.rand(1, 32769, device="mps")
print(torch.mm(x, y[:, 16384:32768]))

import torch.nn as nn
import torch
def print_stats(y, y_p):
    abs_diff = (y - y_p).abs()
    items_different = (abs_diff > 1e-3).int().sum()

    print(
        f"linear diff sum: {abs_diff.sum()}"
    )
    print(
        f"items different (> 1e-3): {items_different} ({100*(int(items_different) / y.numel())}%)"
        )


x_cpu = torch.rand((1025, 1), dtype=torch.float32)
linear_cpu = nn.Linear(in_features=1, out_features=32769,bias=False)
linear_cpu.eval()
y_cpu = linear_cpu(x_cpu)
print(y_cpu.shape)
print("=== CPU matmul ===")
y_matmul = torch.matmul(x_cpu,linear_cpu.weight.t())
print_stats(y_matmul, y_cpu)

x_mps = x_cpu.to("mps")
linear_mps = linear_cpu.to("mps")

print("=== MPS Linear ===")
y_mps = linear_mps(x_mps)
print_stats(y_mps.to("cpu"), y_cpu)

print("=== MPS matmul ===")
y_mps_matmul = torch.matmul(x_mps,linear_mps.weight.t())
print_stats(y_mps_matmul.to("cpu"), y_cpu)