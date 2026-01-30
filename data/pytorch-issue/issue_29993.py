import torch
import torch.nn as nn
import onnx

path = 'converted.onnx'
dummy_input = torch.zeros((1, 3, 32, 32))
zp = nn.ZeroPad2d((0, 1, 0, 1))
torch.onnx.export(zp, dummy_input, path,
                  input_names=['input'], 
                  output_names=['output'],
                  opset_version=11,
                  verbose=True)
# check model
onnx_model = onnx.load(path)
onnx.checker.check_model(onnx_model)