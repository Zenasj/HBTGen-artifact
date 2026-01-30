import numpy as np

import torch

device = 'cpu'
# device = 'cuda'  # This works
dtype = torch.float

inp = torch.randn(10, 2, device=device)
noncontig_input = inp.movedim(-1, 0)

args = (1, 0, 10)

expected = torch.narrow_copy(noncontig_input.contiguous(), 1, 0, 10)
actual = torch.narrow_copy(noncontig_input, 1, 0, 10)

torch.testing.assert_close(actual, expected)