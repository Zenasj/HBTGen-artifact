import torch.nn as nn

#!/usr/bin/env python


import torch
from torch import norm_except_dim
from torch.nn import Linear


class ApplyMask:
    """Hook that applies a mask to a tensor.

    Parameters
    ----------
    norm : bool, optional
        If True, the mask is applied to a norm vector (i.e., g) rather
        than a matrix (i.e., v or w). Default is False.
    """

    def __init__(self, mask, dim=0, norm=False):
        # Precompute the masked indices.
        self._zero_indices = None
        if norm:
            # For g, we need to zet to zero only those vectors
            # that have zero norm because of the mask.
            self._zero_indices = torch.nonzero(norm_except_dim(mask, 2, dim).flatten() == 0.0)
        else:
            self._zero_indices = mask == 0.0

    def __call__(self, w):
        # Hooks are not supposed to modify the argument.
        w = w.clone()
        # A simple element-wise multiplication doesn't work if there are NaNs.
        w[self._zero_indices] = 0.0

        print(f'\nPerformed operation successfuly for tensor of shape {w.shape}\n')
        return w


if __name__ == '__main__':
    batch_size = 2
    in_features = 4
    out_features = 5

    # Generate random input. Make sure the test is reproducible.
    generator = torch.Generator()
    generator.manual_seed(0)
    x = torch.randn(batch_size, in_features, generator=generator, requires_grad=True)

    # Lower triangular mask.
    mask = torch.tril(torch.ones(out_features, in_features, requires_grad=False))

    # Create a weight-normalized masked linear layer.
    linear = Linear(in_features, out_features, bias=True)

    # Apply weight normalization.
    from torch.nn.utils import weight_norm
    linear = weight_norm(linear, name='weight', dim=0)

    # Register hook to zero out gradient in the masked weights.
    linear.weight_g.register_hook(ApplyMask(mask, dim=0, norm=True))
    linear.weight_v.register_hook(ApplyMask(mask))

    # The gradient of the masked parameters should be zero.
    y = linear(x)
    loss = torch.sum(y)
    loss.backward()