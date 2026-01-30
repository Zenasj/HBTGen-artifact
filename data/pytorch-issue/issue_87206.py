import torch

import PIL.Image as Image
import numpy as np

x = torch.randn(1, 4, 2)
y = torch.tensor(np.asarray(Image.new("1", (2, 4), color=1))).unsqueeze(0)
print(type(y), y.shape, y.dtype, y.is_contiguous())  # <class 'torch.Tensor'> torch.Size([1, 4, 2]) torch.bool True

result = torch.masked_select(x, y)
print(result.shape)  # torch.Size([2040])
# The line above must output 8, but instead it outputs 2040.