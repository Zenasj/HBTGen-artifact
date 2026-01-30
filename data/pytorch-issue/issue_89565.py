import torch

# adding two tensors with dimension casts dtype to `float64`
(torch.tensor([1], dtype=torch.float16) + torch.tensor([1], dtype=torch.float64)).dtype
#> torch.float64

# adding a tensor without dimension to a tensor with dimension doesn't cast the dtype to `float64`
(torch.tensor([1], dtype=torch.float16) + torch.tensor(1, dtype=torch.float64)).dtype
#> torch.float16
# expected: torch.float64 or raise an exception
# the same applies for `float32`