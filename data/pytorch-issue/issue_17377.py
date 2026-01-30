import torch
import torch.nn as nn
from torch.onnx import OperatorExportTypes


class AvgPool2dModule(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.aap2d = nn.AdaptiveAvgPool2d(output_size=(output_size, output_size))
    def forward(self, inp):
        return self.aap2d(inp)
        
dummy_input = torch.randn(1, 3, 200, 400)
simple_module = AvgPool2dModule(2)

torch.onnx.export(simple_module, 
                  dummy_input,
                  'simple_module.onnx',
                  verbose=True,
                  operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)