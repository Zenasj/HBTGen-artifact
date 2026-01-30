import torch

input = torch.LongTensor([2])
torch.onnx.export(model, input, "foo.onnx")