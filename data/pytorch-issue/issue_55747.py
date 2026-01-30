import torch.nn as nn

import torch
from torch.nn import Linear,Sequential
class testmodule(torch.nn.Module):
    def __init__(self,nlayer):
        super(testmodule,self).__init__()
        module=[]
        for i in range(nlayer):
            module.append(Linear(20,20))
        self.layers=Sequential(*module)

    def forward(self,x):
        out=self.layers(x)
        return out

class pes(torch.nn.Module):
    def __init__(self):
        super(pes,self).__init__()
        self.module=testmodule(3)

    def forward(self,x):
        out=torch.sum(self.module(x))
        force=torch.autograd.grad([out],[x])[0]
        return out,force

nnmodule=torch.jit.script(pes())
x=torch.ones(10,20)
x.requires_grad=True
out=nnmodule(x)
onnx_module=torch.onnx.export(nnmodule,(x),"test.onnx",input_names=['x'],output_names=['y'],\
example_outputs=out,opset_version=12,dynamic_axes={"x":{0:"batch"},"y":{0:"batch"}})