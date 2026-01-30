import torch.nn as nn

3
import torch

N=1
C_in=1
L_in=1
H_in=3
C_out=1
Ksize=2
Pad=0
Dilation=4
Stride=1
OutPad=2

torch_input = torch.randn(N, C_in, L_in, H_in)
torch_input = torch_input.to("cuda")
torch_input.requires_grad = True
torch_m = torch.nn.ConvTranspose2d(in_channels=C_in, out_channels=C_out, kernel_size=Ksize, padding=Pad, stride=Stride,
    output_padding=(OutPad), dilation=Dilation, bias=False)
torch_m.to("cuda")
torch_out = torch_m(torch_input)
print(torch_out)

import torch

N = 1
C_in = 1
L_in = 1
H_in = 3
C_out = 1
Ksize = 2
Pad = 0
Dilation = 4
Stride = 1
OutPad = 2

torch_input = torch.randn(N, C_in, L_in, H_in)
torch_input.requires_grad = True

torch_m = torch.nn.Conv2d(
    in_channels=C_in, out_channels=C_out,
    kernel_size=Ksize, padding=Pad, stride=Stride,
    dilation=Dilation, bias=False
)
torch_out = torch_m(torch_input)
print(torch_out)