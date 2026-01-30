3
import io
import torch
import torch.nn as nn
import torch.ao.quantization
import torch.nn.quantized
from torch.ao.quantization import DeQuantStub, QuantStub

class SplitConvQuant(nn.Module):
    def __init__(self):
        super(SplitConvQuant, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=1, dilation=1, padding=1, groups=2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, dilation=4, padding=1, groups=2, bias=False)

        self.quant_float = torch.nn.quantized.FloatFunctional()
        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()

    def forward(self, x):
        x = self.quant_stub(x)
        x1, x2 = x.split(32, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = self.quant_float.cat([x1, x2], dim=1)
        x = self.dequant_stub(x)
        return x

3
if __name__ == "__main__":
    model = SplitConvQuant()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")

    x = torch.rand(1, 64, 32, 32)
    # torch.ao.quantization.prepare_qat(model, inplace=True)
    torch.ao.quantization.prepare(model, inplace=True)
    model.eval()

    torch.ao.quantization.convert(model, inplace=True)

    model.eval()
    x = torch.rand(1, 64, 32, 32)
    model(x)

    f = io.BytesIO()

    torch.onnx.export(
        model,
        x,
        f,
        opset_version=16,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
    )