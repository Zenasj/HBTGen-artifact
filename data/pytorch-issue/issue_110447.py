3
# File contents of onnx.py
import torch
import torch.nn as nn
torch.onnx.export(nn.Linear(10, 10), args=torch.ones(10), f='output.onnx')

3
import torch
import torch.nn as nn
torch.onnx.export(nn.Linear(10, 10), args=torch.ones(10), f='output.onnx')

3
print("Hello World")