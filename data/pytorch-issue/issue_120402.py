def linalg_solve_with_flag(A, B, *, left: bool = True, matrix_B: bool = False):
    if matrix_B and B.ndim == 2:
        return torch.vmap(lambda A: torch.linalg.solve(A, B, left=left))(A)
    return torch.linalg.solve(A, B, left=left)

from itertools import product

import torch

def test_linalg_solve(b, n, left):
    A = torch.rand(b, n, n)
    B = torch.rand(n, n)
    return torch.linalg.solve(A, B, left=left)

opts = [16, 32, 64]

# torch.linalg.solve defaults to assuming B is a batch of vectors. There is no way to change this
# behaviour, such as a 'matrix_B=True' flag, so in cases where batch size == n, linalg.solve fails
# in one of two ways, depending on 'left':

def scan_sizes(solve_fn):
    for b, n, left in product(opts, opts, [False, True]):
        msg = f"n = {n}, batch size = {b}, left = {left}"
        expected_shape = torch.Size([b, n, n])
        try:
            out = solve_fn(b, n, left)
        except RuntimeError:
            print(f"❌ RuntimeError for {msg}")
            continue
        if out.shape != expected_shape:
            print(f"❌ Shape {tuple(out.shape)} != {tuple(expected_shape)} for {msg}")
        else:
            print("✅")
    print("\n")

print("Testing standard linalg solve...")
scan_sizes(test_linalg_solve)

def matrix_B_solve(A, B, left):
    return torch.vmap(lambda A: torch.linalg.solve(A, B, left=left))(A)

def test_linalg_vmap_solve(b, n, left):
    A = torch.rand(b, n, n)
    B = torch.rand(n, n)
    return matrix_B_solve(A, B, left=left)

print("Testing vmapped linalg solve...")
scan_sizes(test_linalg_vmap_solve)

def linalg_solve_with_flag(A, B, *, left: bool = True, matrix_B: bool = False):
    if matrix_B and B.ndim == 2:
        return torch.vmap(lambda A: torch.linalg.solve(A, B, left=left))(A)
    return torch.linalg.solve(A, B, left=left)


import time

n = 128
A = torch.rand(n, n, n)
B = torch.rand(n, n)

_ = linalg_solve_with_flag(A, B)
t = time.time()
out = linalg_solve_with_flag(A, B, matrix_B=True)
print(f"Flagged version: {time.time() - t}")

for i in range(n):
    ref = torch.linalg.solve(A[i], B)

    assert torch.all(torch.isclose(out[i], ref)), f"{(out[i] - ref).abs().sum()}"