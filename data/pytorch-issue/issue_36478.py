class MyClass(nn.Module):
    def __init__(self):
        super(MyClass, self).__init__()
    def forward(self, x: List[Tensor]):
        x = x[0]
        return x
model = MyClass()
img = [torch.zeros((2, 3,))]
x = model(img) # OK
torch.onnx.export(model, img, 'f.onnx', verbose=False, opset_version=11, example_outputs=img[0]) # OK
model = torch.jit.script(model)
torch.onnx.export(model, img, 'f.onnx', verbose=False, opset_version=11, example_outputs=img[0]) # fail

import torch
import torch.nn as nn
from torch import Tensor

from typing import List

class MyClass(nn.Module):
    def __init__(self):
        super(MyClass, self).__init__()
    def forward(self, x: List[Tensor]):
        # x = x[0] # FAILS
        x1 = x[0] # WORKS
        return x
        
model = MyClass()
img = [torch.zeros((2, 3,))]
x = model(img) # OK

torch.onnx.export(model, img, 'f.onnx', verbose=False, opset_version=11, example_outputs=img[0]) # OK
model = torch.jit.script(model)
torch.onnx.export(model, img, 'f.onnx', verbose=False, opset_version=11, example_outputs=img[0]) # fail