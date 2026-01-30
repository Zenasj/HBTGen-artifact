import torch
import torch.nn as nn

deconv = nn.ConvTranspose2d(in_channels=256, out_channels=128,
           kernel_size=3, stride=2, padding=1, output_padding=1).cuda().half()

batch = torch.randn(8,256,16,16).cuda().half()
output = deconv(batch)
gO = torch.rand_like(output)
output.backward(gO)