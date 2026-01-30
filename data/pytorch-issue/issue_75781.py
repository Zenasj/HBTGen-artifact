import torch

if __name__ == "__main__":

    n = 8
    x = torch.zeros(n).normal_()
    x.requires_grad = True
    z = torch.fft.irfft(x).sum()
    z.backward()

import torch

func_cls=torch.fft.ifftn

if __name__ == "__main__":

    n = 8
    x = torch.zeros(n).normal_()
    x.requires_grad = True
    z = func_cls(x).sum()
    z.backward()