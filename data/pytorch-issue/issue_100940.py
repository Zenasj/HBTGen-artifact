import torch.nn as nn

py
import torch

torch.manual_seed(420)

x = torch.randn(3, 3, 16, 16)

class Module(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(2, 1), padding=(0,), dilation=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding=(3,), dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.mul(x, x)
        x = self.conv2(x)
        return x



func = Module().to('cpu')


with torch.no_grad():
    func.train(False)

    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # torch._C._nn.mkldnn_reorder_conv2d_weight(
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # RuntimeError: dimensions are invalid