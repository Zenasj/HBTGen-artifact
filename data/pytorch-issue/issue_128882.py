import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # fails with:  TypeError: _functionalize_sync(): argument 't' (position 1) must be Tensor, not SymInt                           
        return x.numel()
                                                                         

device = torch.device('cuda')
model = Model().to(device)
x = torch.rand(1024, 20, 16).to(device)


onnx_program = torch.onnx.export(model, x, "legacy_numel.onnx", input_names=["x"], dynamic_axes={"x":[0,1,2]}, verbose=True)

# This does not help                                                                                                                             
# model = torch.export.export(model, (x,), strict=False).run_decompositions()                                                                    
options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(model, x, export_options=options)
onnx_program.save('numel.onnx')