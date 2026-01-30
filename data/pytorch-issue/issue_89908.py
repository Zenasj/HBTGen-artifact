import torch
import torch.nn as nn

class SomeOp(torch.nn.Module):
    def __init__(self):
        super(SomeOp, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2:]
        res = torch.cat([y[:, 0:1, :, :] / width, y[:, 1:2, :, :] / height], 1)
        return res


class TestNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(2, 2, 1)
        self.some_op = SomeOp()

    def forward(self, x, y):
        return self.some_op(self.some_op(x, y), y)

net = TestNet()
x = torch.rand((1, 2, 16, 16))
y = torch.rand((1, 2, 16, 16))
onnx_path = os.path.join(this_dir, 'test.onnx')
torch.onnx.export(net, (x, y), onnx_path,
                      verbose=True, opset_version=16,
                      export_modules_as_functions={SomeOp})