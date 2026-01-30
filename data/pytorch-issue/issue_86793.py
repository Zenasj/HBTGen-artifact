import torch
import torchvision.transforms.functional as T

@torch.jit.script
def blur(x: torch.Tensor, sigma: float=2.0) -> torch.Tensor:
    k = int(round(4.0 * sigma))
    if k % 2 == 0:
        k += 1
    x = T.gaussian_blur(x, [k, k], [sigma, sigma])
    return x

x = torch.rand(3, 16, 16)
blur(x, 1.0)
blur(x, 2.0)

print(blur.code)

print(blur.graph)