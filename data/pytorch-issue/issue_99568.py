py
import torch
import torch.nn as nn

torch.manual_seed(420)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)

    def forward(self, x):
        x = self.conv(x)
        return x

input_tensor = torch.randn(3, 32, 32)

func = Model().to('cpu')

print(func(input_tensor))
# Success

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    print(jit_func(input_tensor))
# RuntimeError: could not create a descriptor for a dilated convolution forward propagation primitive