import torch
import sys

i = sys.maxsize + 1

input = torch.full((1, 32, 32,), 0.5)
torch.max_pool1d(input, kernel_size=[i] , stride=[i], padding=0, dilation=[i], ceil_mode=True)