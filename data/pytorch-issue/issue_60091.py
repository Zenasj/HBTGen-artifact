import torch

torch.testing.assert_close(1, 3, rtol=0, atol=1)

torch.manual_seed(0)
t = torch.rand((2, 2), dtype=torch.complex64)
torch.testing.assert_close(t, t + complex(0, 1))