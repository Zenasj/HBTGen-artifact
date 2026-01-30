import torch.nn as nn

py
import torch


class FakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, axis):
        return (x / scale).round().clamp(-127, 127) * scale

    @staticmethod
    def symbolic(g, x, scale, axis):
        zero_point = g.op("Constant", value_t=torch.zeros(1, dtype=torch.int32))
        quant = g.op("Horizon::QuantizeLinear", x, scale, zero_point, axis_i=axis).setType(x.type())
        # return quant
        dequant = g.op("Horizon::DeQuantizeLinear", quant, scale, zero_point, axis_i=axis).setType(x.type())
        return dequant


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        weight = torch.randn(1, 3, 1, 1)
        weight = FakeQuantizeFunction.apply(weight, torch.ones(1), 0)
        return torch.nn.functional.conv2d(input=x, weight=weight, bias=None)


with torch.no_grad():
    net = Net()
    net.eval()
    x = torch.zeros(1, 3, 10, 10)
    print(net(x))
    onnx = torch.onnx.export(net, x, "test.onnx", verbose=True)