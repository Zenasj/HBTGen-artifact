import torch.nn as nn

import torch

dummy_input = (torch.tensor([1, 4, 2, 7, 3]), torch.tensor([1, 2, 2]))

class Split(torch.nn.Module):
    def forward(self, x, l):
        return x.split(l.cpu().numpy().tolist(), dim=-1)
    
model = Split()

with torch.no_grad():
    torch.onnx.export(
        model, dummy_input, 'split.onnx', verbose=False, opset_version=13,
        input_names=['a', 'b'],
        output_names=['c'],
        dynamic_axes={'a': [0], 'b': [0], 'c': [0]}
    )

import onnxruntime as ort

model_path = './split.onnx'
sess = ort.InferenceSession(model_path)

a = torch.tensor([4, 2, 3, 4])
b = torch.tensor([1, 3])
sess.run(['c'], {'a':a.numpy(), 'b':b.numpy()})

import torch
import onnxruntime as ort

dummy_input = (torch.tensor([1, 4, 2, 7, 3]), [torch.tensor(1), torch.tensor(2), torch.tensor(2)])

class Split(torch.nn.Module):
    def forward(self, x, l):
        return x.split(l)
    
model = Split()

with torch.no_grad():
    torch.onnx.export(
        model, dummy_input, 'split.onnx', verbose=False, opset_version=13,
        input_names=['a', 'b'],
        output_names=['c'],
        dynamic_axes={'a': [0], 'c': [0]}
    )

model_path = './split.onnx'
sess = ort.InferenceSession(model_path)

a = torch.tensor([4, 2, 3, 4])
b = [torch.tensor(1).numpy(), torch.tensor(3).numpy(), torch.tensor(1).numpy()]
sess.run(['c'], {'a':a.numpy(), 'b':b})