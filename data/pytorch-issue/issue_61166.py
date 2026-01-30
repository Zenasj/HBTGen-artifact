import torch
import torch.nn as nn
import numpy as np

input = torch.tensor(
        np.arange(1, 5, dtype=np.int32).reshape((1, 1, 2, 2)) )
m = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
of_out = m(input)