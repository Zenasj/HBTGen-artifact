import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = torch.cat([out, out], 1)
        return out

model = TestModel(3,1)

model.eval()
qconfig = torch.quantization.get_default_qconfig('fbgemm')
model.qconfig = qconfig

# Fuse modules here if required

model = nn.Sequential(torch.quantization.QuantStub(qconfig), 
                      model, 
                      torch.quantization.DeQuantStub(qconfig))

torch.quantization.prepare(model, inplace=True)

with torch.inference_mode():
    for _ in range(100):
        model(torch.randn(1,3,32,32))

torch.quantization.convert(model, inplace=True)

model.eval()
qconfig_dict = {"": torch.quantization.get_default_qconfig('fbgemm')}
model = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs=(torch.randn(1,3,32,32),))

with torch.inference_mode():
    for _ in range(100):
        model(torch.randn(1,3,32,32))

model = quantize_fx.convert_fx(model)