import torch

@torch.jit.script
def foo(input: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]):
    changed_input = input[0] + 1
    return (changed_input,) + input[1:]  # not supported in JIT but is supported in eager