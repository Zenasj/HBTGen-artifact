import torch.nn as nn

import torch

class DummyModule(torch.nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
        
    def forward(self, x, y):
        for i in range(y.size(0)):
            x = x + i
        return x

# Instantiation and scripting
model_scripted = torch.jit.script(DummyModule())
dummy_input_one = torch.Tensor([1,2,3]).float()
dummy_input_two = torch.Tensor([4,5,6]).long()

# Check if the forward pass works:
output = model_scripted(dummy_input_one, dummy_input_two)
print(output)

# Export to onnx:
torch.onnx.export(model_scripted, 
                  (dummy_input_one, dummy_input_two), 
                  'loop.onnx', 
                  verbose=True,
                  input_names=['input_data', 'input_data_2'], 
                  example_outputs=output
                 )

# Load the model in onnx runtime:
import onnxruntime as rt

sess = rt.InferenceSession("loop.onnx")

import onnx
import caffe2.python.onnx.backend as backend

model = onnx.load('loop.onnx')

model_runtime = backend.prepare(model)

x = x + i