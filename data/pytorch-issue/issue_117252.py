import torch

self = torch.randint((1 << 15) - 1, [1,1,1,1,1], dtype=torch.int32)
torch.fft.fft2(self, None, [], None)