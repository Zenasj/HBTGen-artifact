import torch

torch.testing.assert_close(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))