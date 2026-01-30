import torch

t = torch.linspace(0, 1, 5, device="cpu")
T = torch.fft.rfft(t)
roundtrip = torch.fft.irfft(T, t.numel())
print(t, roundtrip)
torch.testing.assert_close(roundtrip, t, check_stride=False)

t = torch.linspace(0, 1, 5, device="mps")
T = torch.fft.rfft(t)
roundtrip = torch.fft.irfft(T, t.numel())
print(t, roundtrip)
torch.testing.assert_close(roundtrip, t, check_stride=False)