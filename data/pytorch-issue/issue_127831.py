import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sort(x, dim=-1, descending=False)

device = torch.device('cuda')
model = Model().to(device)
x = torch.rand(1024, 20, 16).to(device)

# This does not help                                                                                                                                      
# model = torch.export.export(model, (x,), strict=False).run_decompositions()                                                                             

# This works 
onnx_program = torch.onnx.export(model, x, "legacy_sort.onnx")
 
options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(model, x, export_options=options)
onnx_program.save('model.onnx')