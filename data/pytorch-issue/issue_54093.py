import torch.nn as nn

import torch

from torch.autograd import gradgradcheck


def stft_with_abs(tensor):
    tensor = torch.stft(input=tensor, n_fft=256, return_complex=True)
    tensor = tensor.abs()
    return tensor


def abs_(tensor):
    return tensor.abs()


def stft(tensor):
    return torch.stft(tensor, n_fft=256, return_complex=True)


def test_stft_with_abs():
    for i in range(100):
        print(i, '\r', end='')
        tensor = torch.randn([2, 250])
        tensor.requires_grad = True

        tensor = tensor.to(dtype=torch.float64, device='cuda')
        assert gradgradcheck(stft_with_abs, [tensor])


def test_stft_only():
    for i in range(100):
        print(i, '\r', end='')
        tensor = torch.randn([2, 250])
        tensor.requires_grad = True

        tensor = tensor.to(dtype=torch.float64, device='cuda')
        assert gradgradcheck(stft, [tensor])


def test_abs_only():
    for i in range(100):
        print(i, '\r', end='')
        tensor = torch.randn([2, 250])
        tensor = tensor.to(dtype=torch.float64, device='cuda')
        tensor = torch.stft(input=tensor, n_fft=256, return_complex=True)

        tensor.requires_grad = True
        assert gradgradcheck(abs_, [tensor])


# test_stft_only()  # does not fail
# test_abs_only()  # does not fail
test_stft_with_abs()

def pad_complex_abs(tensor):
    tensor = torch.nn.functional.pad(tensor, [128, 128], 'reflect')
    tensor = tensor.transpose(0, -1).contiguous()
    tensor = torch.view_as_complex(tensor)
    tensor = torch.abs(tensor)
    return tensor

for _ in range(100):
    tensor = torch.randn(2, 1, 250, dtype=torch.float64, device='cuda', requires_grad=True)
    gradgradcheck(pad_complex_abs, [tensor])