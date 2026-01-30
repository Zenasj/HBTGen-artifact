import torch

shell
net = SegNet()
input = torch.randn(1,3,224,224)
torch.onnx.export(
    net,                  # pytorch model
    input,      # input (tuple for multiple inputes)
    "segnet.onnx",          # onnx model to save
    export_params=True,        # store the trained parameter weights inside the model file
    opset_version=13,          # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=['input'],   # the model's input names
    output_names=['output'],  # the model's output names
)