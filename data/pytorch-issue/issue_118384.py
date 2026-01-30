import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

i, j, k = 744, 744, 16
m = 31
l = 100

# Initialise Layer and Input Tensor
conv_layer = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=m, stride=1, padding=15, padding_mode="circular", bias=False)
conv_layer.to(device)

conv_layer.weight.data = torch.randn(k, k, m, m, device=device)

input_tensor = torch.randn(l, k, i, j, device=device)

# Loop for convolutions
for x in range(300): # Problem Around Here
    output = conv_layer(input_tensor)
    del output
    print(x)