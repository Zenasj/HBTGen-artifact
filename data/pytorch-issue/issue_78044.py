import torch
x = torch.randn(1, 16000, device="mps")
y = torch.fft.rfft(x)
y_abs = y.abs()

x = torch.randn(1, 16000, device="cpu")
y = torch.fft.rfft(x)
y_abs = y.abs()