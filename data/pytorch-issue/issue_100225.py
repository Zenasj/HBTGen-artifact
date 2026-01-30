import torch.nn as nn

py
import torch

torch.manual_seed(420)

x = torch.randn(1, 3, 28, 28)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 6, 3, stride=1, padding=1, output_padding=1)

    def forward(self, input_tensor):
        x = self.conv_transpose(input_tensor)
        output = torch.tanh(x)
        return output

func = Model().to('cpu')


with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # Success

    res1 = func(x)
    # RuntimeError: output padding must be smaller than either stride or dilation, but got output_padding_height: 1 output_padding_width: 1 stride_height: 1 stride_width: 1 dilation_height: 1 dilation_width: 1