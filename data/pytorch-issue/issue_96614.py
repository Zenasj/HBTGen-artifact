import torch

input = torch.randn(1000,2) < 0.5
mps_result = input.to('mps').cumsum(0)
cpu_result = input.cumsum(0)
torch.testing.assert_close(mps_result, cpu_result, check_device=False)