import torch

# make a quantized tensor
r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
scale = 0.02
zero_point = 2
quantized = torch.quantize_per_tensor(r, scale, zero_point, torch.quint8)

# The following invokes the quantized key's version of the clamp.
# The quantized key doesn't actually have a kernel registered for clamp,
# so it throws an error to us.
torch.clamp(quantized, quantized, quantized)

# However, the following straight up calls into native::clamp (e.g. the CPU
# kernel) instead of dispatching on the quantized key and giving us an error:
cpu_tensor = torch.randn(3, 2)
torch.clamp(cpu_tensor, quantized, quantized)