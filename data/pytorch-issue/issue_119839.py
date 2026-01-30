import torch

if schema.is_mutable and not can_auto_functionalize(kernel):
    raise NotImplementedError(
        f"NYI: Can't generate FallbackKernel for {kernel}"
    )

def _all_reduce_single(input: torch.Tensor, op: str):
    res = torch.ops._c10d_functional.all_reduce(input.contiguous(), op, "default")
    return torch.ops._c10d_functional.wait_tensor(res)