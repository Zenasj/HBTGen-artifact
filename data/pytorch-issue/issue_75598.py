import torch

input_names = ['Input']
output_names = ['Output']
tokens = torch.randint(0, 20000, (1, 2048))
torch.onnx.export(model, tokens, 'model.onnx', input_names=input_names, output_names=output_names, opset_version=12)