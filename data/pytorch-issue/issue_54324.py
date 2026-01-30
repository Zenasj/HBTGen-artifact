import torch
import torch.nn as nn


dtype = torch.float16
device = torch.device('cuda:0')
num_channels = 64
padding = 1
groups = 2
input_size = 1000

model = nn.Sequential(
    nn.Conv2d(1, num_channels, 3),
    nn.Conv2d(num_channels, num_channels, 3, padding=padding, groups=groups),
)   
model = model.to(dtype=dtype, device=device)
input = torch.randn([1, 1, input_size, input_size], dtype=dtype, device=device)
_ = model(input)