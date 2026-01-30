import torch

assert torch.eq(actual, expected), "Tensors are not equal!"
torch.testing.assert_equal(actual, expected)