import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv_transpose_output = self.conv_transpose1(x)
        clamp_min_output = torch.clamp_min(self.conv_transpose2(conv_transpose_output), 3)
        clamp_max_output = torch.clamp_max(clamp_min_output, 0)
        div_output = torch.div(clamp_max_output, 6)
        return div_output

x = torch.randn((1, 3, 128, 128))

func = Model()

res1 = func(x)
print(res1)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # RuntimeError: could not append an elementwise post-op
    # buf2 = torch.ops.mkldnn._convolution_transpose_pointwise(buf1, arg2_1, arg3_1, (1, 1), (0, 0), (1, 1), (1, 1), 1, 'hardtanh', [3, 0], '')