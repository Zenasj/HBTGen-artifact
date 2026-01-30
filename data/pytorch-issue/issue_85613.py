from torch import randn, contiguous_format
from einops import rearrange

x = randn(1, 512, 512, 3, device='cpu')
x = rearrange(x, 'b h w c -> b c h w')
y = x.to(memory_format=contiguous_format)
print(y.to('mps').cpu().allclose(y))
# False (and bad)
z = x.contiguous()
print(z.to('mps').cpu().allclose(z))
# True