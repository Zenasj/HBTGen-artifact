import torch
import torch.nn as nn

norm = nn.GroupNorm(8, 32).to(device, memory_format=torch.channels_last)
x = torch.randn([4, 32, 24, 24], device=device).to(memory_format=torch.channels_last)

print(x.stride())
assert x.is_contiguous(memory_format=torch.channels_last) # Pass

y = norm(x)

print(y.stride())
assert y.is_contiguous(memory_format=torch.channels_last) # Fail

class GroupNorm(nn.GroupNorm):
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = rearrange(x, "b (g c) h w -> b g c h w", g=self.num_groups)

        mean = x.mean(dim=[2,3,4], keepdim=True)
        var = x.var(dim=[2,3,4], keepdim=True)

        x = (x - mean) * (var + self.eps).rsqrt()
        x = rearrange(x, "b g c h w -> b (g c) h w")

        if self.affine:
            weight = rearrange(self.weight, "c -> 1 c 1 1")
            bias = rearrange(self.bias, "c -> 1 c 1 1")
            x = x * weight + bias

        x = x.type(dtype)
        return x