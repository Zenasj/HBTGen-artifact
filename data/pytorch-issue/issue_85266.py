import torch
import torch.nn as nn

rfft_input = torch.nn.functional.pad(y.view(1, -1), [50, 50], mode='reflect').squeeze()[:100]
output3 = torch.fft.rfft(rfft_input)
assert torch.allclose(output2, output3)