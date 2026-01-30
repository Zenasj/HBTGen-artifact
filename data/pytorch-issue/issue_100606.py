import torch
import torch._inductor

torch._inductor.config.disable_cpp_codegen = True

@torch.compile(backend="inductor", dynamic=True)
def func(input: torch.Tensor) -> torch.Tensor:
    n = input.size(-1)
    output = input + int(n * 0.2) + 1
    return output, input + 1


print(func(torch.rand(5, device="cpu")))
print(func(torch.rand(10, device="cpu")))