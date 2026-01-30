import torch
from monai.networks.nets import FullyConnectedNet

model = lambda : FullyConnectedNet(
    10, #inSize
    3, #outSize
    [8, 16], #channels
    0.15
)
model = model().eval().to('cuda')
data = torch.randn(4, 10).to("cuda")

dynamo_export = True
if dynamo_export:
    model = torch.export.export(model, args=(data,))
    export_output = torch.onnx.dynamo_export(model, data)
    export_output.save('Clara_FullyConnectedNet_dynamo.onnx')
else:
    torch.onnx.export(model, (data,), 'Clara_FullConnectedNet_torchscript.onnx')

model = torch.export.export(model, args=(data,))
export_output = torch.onnx.dynamo_export(model, data)
export_output.save('Clara_FullyConnectedNet_dynamo.onnx', model_state=model.state_dict)

import torch_onnx
torch_onnx.patch_torch()

import torch
from monai.networks.nets import FullyConnectedNet


model = lambda : FullyConnectedNet(
    10, #inSize
    3, #outSize
    [8, 16], #channels
    0.15
)
data = torch.randn(4, 10).to("cuda")
model = model().eval().to('cuda')

import torch_onnx
torch_onnx.patch_torch()
model = torch.export.export(model, args=(data,))
export_output = torch.onnx.dynamo_export(model, data)
export_output.save('Clara_FullyConnectedNet_dynamo.onnx', model_state=model.state_dict)