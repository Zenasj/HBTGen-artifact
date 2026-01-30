import numpy as np
import torch

t = torch.arange(0, 7, 1)
print(torch.fft.rfftn(t).shape)
print(np.fft.rfftn(t).shape)

torch.Size([4])
(4,)