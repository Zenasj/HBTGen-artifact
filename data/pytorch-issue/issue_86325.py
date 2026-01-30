import torch.nn as nn

import torch

class Inner(torch.nn.Module):
    def forward(self, x):
        if x > 0 :  
            return x
        else:
            return x*x

class Outer(torch.nn.Module):
    def __init__(self):   
        super().__init__()
        i = Inner()
        self.inner = torch.jit.script(i)

    def forward(self, x):   
        return self.inner(x)

x = torch.zeros(1)
o=Outer()
o.eval()
m = torch.jit.trace_module(o, { 'forward' : (x)})
# borisf: passes if you comment this line out                                                                                                                                                        
m = torch.jit.optimize_for_inference(torch.jit.freeze(m))

torch.onnx.export(m, (x,), 'test.onnx')

import torch

# loading your TorchScript
model = torch.jit.load("model.pt")

# converting the model to ONNX
dummy_input = ...
torch.onnx.export(model, dummy_input, "model.onnx")

import torch

# loading your TorchScript
model = torch.jit.load("model.pt")

# ADD THIS LINE
model.training = False

# converting the model to ONNX
dummy_input = ...
torch.onnx.export(model, dummy_input, "model.onnx")