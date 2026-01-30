import torch

model.train(False)
input = torch.randn(1, 3, 224, 224, requires_grad=True)
output = model(input)
torch_out = torch.onnx._export(model, input, "model.onnx", export_params=True, do_constant_folding=True)