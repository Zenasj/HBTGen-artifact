import torch.nn.functional as F

...

MAX_ELEMENTS_FOR_BACKPROP = int(2e8)

bond_losses = F.mse_loss(denoised_cdist, normalized_cdist, reduction = 'none')
bond_losses = bond_losses * loss_weights

if atompair_mask.sum() > MAX_ELEMENTS_FOR_BACKPROP:
      # randomly subset the atom pairs to supervise to avoid HIP error during backprop
      
      flat_atompair_mask_indices = torch.arange(atompair_mask.numel(), device=self.device)[atompair_mask.view(-1)]
      num_true_atompairs = flat_atompair_mask_indices.size(0)
      
      num_atompairs_to_ignore = num_true_atompairs - MAX_ELEMENTS_FOR_BACKPROP
      ignored_atompair_indices = flat_atompair_mask_indices[torch.randperm(num_true_atompairs)[:num_atompairs_to_ignore]]
      
      atompair_mask.view(-1)[ignored_atompair_indices] = False

bond_loss = bond_losses[atompair_mask].mean()  # <- without the if-clause above, this is the error-triggering line for large inputs

# in my nn.Module's init function...
from einops.layers.torch import Rearrange

dim_pairwise = 16
heads = 8

LinearNoBias = partial(nn.Linear, bias=False)
to_attn_bias_linear = LinearNoBias(dim_pairwise, heads)
nn.init.zeros_(to_attn_bias_linear.weight)

self.to_attn_bias_norm = nn.LayerNorm(dim_pairwise)
self.to_attn_bias = nn.Sequential(to_attn_bias_linear, Rearrange("b ... h -> b h ..."))

# in my nn.Module's forward function...
MAX_CONCURRENT_TENSOR_ELEMENTS = int(1e8)  # my workaround
b, n, dp = pairwise_repr.shape[0], pairwise_repr.shape[1], pairwise_repr.shape[-1]
dtype, device = pairwise_repr.dtype, pairwise_repr.device

if pairwise_repr.numel() > MAX_CONCURRENT_TENSOR_ELEMENTS:
    # create a stub tensor and normalize it to maintain gradients to `to_attn_bias_norm`
    stub_pairwise_repr = torch.zeros((b, dp), dtype=dtype, device=device)
    stub_attn_bias_norm = self.to_attn_bias_norm(stub_pairwise_repr) * 0.0
    
    # adjust `attn_bias_norm` dimensions to match `pairwise_repr`
    attn_bias_norm = pairwise_repr + stub_attn_bias_norm[:, None, None, :]
    
    # apply bias transformation
    attn_bias = self.to_attn_bias(attn_bias_norm) + attn_bias
else:
    attn_bias = self.to_attn_bias(self.to_attn_bias_norm(pairwise_repr)) + attn_bias

from functools import partial

import torch
import torch.nn as nn

MAX_CONCURRENT_TENSOR_ELEMENTS = int(1e20)  # NOTE: set to 2e9 for my workaround


class ReproducePyTorchIssue136291(nn.Module):
    """Reproduce PyTorch issue #136291."""

    def __init__(
        self,
        b=16,
        n=3000,
        dp=16,
        heads=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        super(ReproducePyTorchIssue136291, self).__init__()
        self.b = b
        self.n = n
        self.dp = dp
        self.heads = heads
        self.dtype = dtype
        self.device = device

        LinearNoBias = partial(nn.Linear, bias=False)
        self.to_attn_bias_linear = LinearNoBias(dp, heads).to(device=device)
        nn.init.zeros_(self.to_attn_bias_linear.weight)

        self.to_attn_bias_norm = nn.LayerNorm(dp).to(device=device)
        self.to_attn_bias = self.to_attn_bias_linear

    def forward(self, pairwise_repr):
        pairwise_repr = pairwise_repr.to(dtype=self.dtype, device=self.device)

        if pairwise_repr.numel() > MAX_CONCURRENT_TENSOR_ELEMENTS:
            stub_pairwise_repr = torch.zeros(
                (self.b, self.dp), dtype=self.dtype, device=self.device
            )
            stub_attn_bias_norm = self.to_attn_bias_norm(stub_pairwise_repr) * 0.0

            attn_bias_norm = pairwise_repr + stub_attn_bias_norm[:, None, None, :]

            attn_bias = self.to_attn_bias(attn_bias_norm).view(self.b, self.heads, self.n, self.n)
        else:
            attn_bias = self.to_attn_bias(self.to_attn_bias_norm(pairwise_repr)).view(
                self.b, self.heads, self.n, self.n
            )

        return attn_bias


if __name__ == "__main__":
    model = ReproducePyTorchIssue136291()
    # NOTE: This issue doesn't occur with `n=2000`.
    b, n, dp = 16, 3000, 16
    pairwise_repr = torch.randn(b, n, n, dp, device="cuda")
    output = model(pairwise_repr)

    print(output.shape)