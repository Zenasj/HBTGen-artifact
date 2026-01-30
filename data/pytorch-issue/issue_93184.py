import torch

cuda_only_fp_list = [
    torch.rand((1, 2), device="cuda", dtype=torch.float32),
    torch.rand((1, 2), device="cuda", dtype=torch.float64),
    torch.rand((1, 2), device="cuda", dtype=torch.float16),
    torch.rand((1, 2), device="cuda", dtype=torch.bfloat16),
]

cuda_only_int_list = [
    torch.randint(1024, (1, 2), device="cuda", dtype=torch.int64),
]

cpu_list = [
    torch.rand((1, 2), device="cpu", dtype=torch.float32),
    torch.rand((1, 2), device="cpu", dtype=torch.float64),
    torch.rand((1, 2), device="cpu", dtype=torch.float16),
]

none_list = [None]

# differentiable should always make it return false for both
assert _default_to_fused_or_foreach([cuda_only_fp_list], True, True) == (False, False)
assert _default_to_fused_or_foreach([cuda_only_fp_list], True, False) == (False, False)

# cpu lists should always make it return false for both
assert _default_to_fused_or_foreach([cuda_only_fp_list, cpu_list], False, True) == (False, False)
assert _default_to_fused_or_foreach([cpu_list], False, True) == (False, False)
assert _default_to_fused_or_foreach([cuda_only_fp_list, cpu_list], False, False) == (False, False)
assert _default_to_fused_or_foreach([cpu_list], False, False) == (False, False)

# has fused triggers correctly
assert _default_to_fused_or_foreach([cuda_only_fp_list], False, True) == (True, False)
assert _default_to_fused_or_foreach([cuda_only_fp_list], False, False) == (False, True)

# ints always goes to foreach
assert _default_to_fused_or_foreach([cuda_only_fp_list, cuda_only_int_list], False, True) == (False, True)
assert _default_to_fused_or_foreach([cuda_only_fp_list, cuda_only_int_list], False, False) == (False, True)

# Nones don't error
assert _default_to_fused_or_foreach([cuda_only_fp_list, none_list], False, True) == (True, False)
assert _default_to_fused_or_foreach([cuda_only_fp_list, cuda_only_int_list, none_list], False, True) == (False, True)
assert _default_to_fused_or_foreach([none_list], False, True) == (True, False)
assert _default_to_fused_or_foreach([none_list], False, False) == (False, True)