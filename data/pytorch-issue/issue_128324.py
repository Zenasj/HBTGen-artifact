import torch
import torch.nn as nn

class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.fft.fftn(x, dim=(-2,-1))
        return x

x = torch.randn(batch_size, 3, 224, 224)
test = Test()
t_out = test(x)
onnx_program_test = torch.onnx.dynamo_export(test, x, export_options=torch.onnx.ExportOptions(dynamic_shapes=True))