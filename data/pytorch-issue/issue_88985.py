import torch

def test():
    arg_1 = torch.rand([], dtype=torch.float64).clone()
    res = torch.fft.hfft(arg_1,1)

test()