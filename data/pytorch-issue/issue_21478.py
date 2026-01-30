@torch.jit.script
def f():
  torch.stft(...)

import torch

@torch.jit.script
def test_stft(input, n_fft):
    # type: (Tensor, int) -> Tensor
    return torch.stft(input, n_fft)

print(test_stft.graph)

inp = torch.randn(4, 5)
res = test_stft(inp, 4)

print(res)

import torch

input = torch.rand((1,100))
n_fft = 20
hop_length=None
win_length=None
window=None
center=True
pad_mode='reflect'
normalized=False
onesided=True

output = torch.stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided)
print(output.size()) # prints torch.Size([1, 11, 21, 2])


@torch.jit.script
def foo(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided):
	# type: (Tensor, int, Optional[int], Optional[int], Optional[Tensor], bool, str, bool, bool) -> Tensor
	output = torch.stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided)
	return output

output = foo(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided)
print(output.size())