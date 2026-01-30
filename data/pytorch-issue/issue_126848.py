import torch
import torch._inductor.config

torch._inductor.config.force_disable_caches = True  # doesn't seem to help

print(torch.__version__)

def func(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 ** (-127 + 3)

x = torch.tensor(0.1875)
print("Eager (before):", func(x))
print("Compile:", torch.compile(func)(x))
print("Eager (after):", func(x))

import torch

print(torch.__version__)

def func(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 ** (-127 + 3)

x = torch.tensor(0.1875)
print("Eager (before):", func(x))

func_compiled = torch.compile(func)
torch.set_flush_denormal(False)
print("Compile:", func_compiled(x))

print("Eager (after 1):", func(x))
torch.set_flush_denormal(False)
print("Eager (after 2):", func(x))

# no rounding logic for brevity
def fp32_to_fp6(tensor: Tensor) -> Tensor:
    # correct exponent bias. this also handles subnormal numbers correctly
    tensor = tensor * 2.0 ** (-127 + 3)
    bits = tensor.view(torch.int32)

    sign = ((bits >> 31) & 0x1) << 5
    exp_and_man = (bits >> 21) & 0x1F
    result = sign | exp_and_man
    return result.to(torch.uint8)  # stored as uint8 data

def fp6_to_fp32(tensor: Tensor) -> Tensor:
    bits = tensor.to(torch.int32)  # bit extension
    sign = bits >> 5 << 31
    exp_and_man = (bits & 0x1F) << 21
    results = sign | exp_and_man

    results = results.view(torch.float32)
    return results * 2.0 ** (127 - 3)  # exponent bias correction