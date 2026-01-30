import torch

@register_decomposition(aten._rms_norm_fused)
def rms_norm_fused(
    self: torch.Tensor, ndim: int, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    dtr = [self.dim() - i - 1 for i in range(ndim)]
    return self * weight * (self.pow(2).mean(dtr, keepdim=True).add(eps).rsqrt())