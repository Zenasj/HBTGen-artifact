import torch
import torch.nn as nn

hidden_dim = 256
kernel_size = 769
stride = 64
padding = kernel_size // 2
samples = torch.randn([25, 1, 240640])

# Zeros padding mode would work
conv1d_zeros = nn.Conv1d(
            1,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode='zeros',
        )
res1 = conv1d_zeros(samples)
print(res1.size())

# result:
torch.Size([25, 256, 3760])

# Padding mode other than zeros would crash with "code is too big"
conv1d_reflect = nn.Conv1d(
            1,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode='reflect',
        )
res2 = conv1d_reflect(samples)
print(res2.size())

# padding with zero padding size would crash with "code is too big"
conv1d_zeros_no_pad = nn.Conv1d(
            1,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            padding_mode='zeros',
        )
res3 = conv1d_zeros_no_pad(samples)
print(res3.size())

n_dim = 205794  # 205512 works, 205794 error
out = torch.nn.functional.conv1d(
    torch.zeros([1, 1, n_dim]), torch.zeros([514, 1, 512]), stride=160, padding=0
)