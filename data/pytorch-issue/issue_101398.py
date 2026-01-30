import torch

dummy_input = torch.ones((1, num_value)).to(device)
with torch.no_grad():
    torch.onnx.export(model, dummy_input, fileName, opset_version=12)

torch.onnx.export(model,...)

torch.onnx.export( torch.jit.trace(model), ... )