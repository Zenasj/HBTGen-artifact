import torch.nn as nn

import torch

@torch.compile
def avg_pool1d(input, kernel_size, stride=None, padding=0):
    return torch.nn.functional.avg_pool1d(input, kernel_size, stride, padding)

input = torch.tensor([[1.7641]])
kernel_size = 4
stride = 3
padding = 2

input = input.cuda()
print(f"[CUDA] AvgPool1d in compiled mode: {avg_pool1d(input, kernel_size, stride, padding)}")
print(f"[CUDA] AvgPool1d in eager mode: {torch.nn.functional.avg_pool1d(input, kernel_size, stride, padding)}")
input = input.cpu()
print(f"[CPU] AvgPool1d in compiled mode: {avg_pool1d(input, kernel_size, stride, padding)}")
print(f"[CPU] AvgPool1d in eager mode: {torch.nn.functional.avg_pool1d(input, kernel_size, stride, padding)}")