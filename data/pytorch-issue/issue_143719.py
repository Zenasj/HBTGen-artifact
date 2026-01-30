import torch

print(torch.__version__)

@torch.compile
def fft(input, n=None, dim=-1, norm=None):
    return torch.fft.fft(input, n, dim, norm)

input = torch.tensor([[[1.3703]]])
input = input.to('cuda')
n = 2
dim = -1
print(f"[CUDA] FFT in compiled mode: {fft(input, n, dim)}")
print(f"[CUDA] FFT in eager mode: {torch.fft.fft(input, n, dim)}")
input = input.cpu()
print(f"[CPU] FFT in compiled mode: {fft(input, n, dim)}")
print(f"[CPU] FFT in eager mode: {torch.fft.fft(input, n, dim)}")