import torch

torch.manual_seed(42)
tens = 0.5 * torch.randn(10, 3, 3, dtype=torch.complex64)
tens = (0.5 * (tens.transpose(-1, -2) + tens))
tens.imag = torch.matrix_exp(tens.imag)