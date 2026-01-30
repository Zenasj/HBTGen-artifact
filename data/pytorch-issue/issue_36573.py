import torch.nn as nn

class MyConvBlockFunction(Function):

    @staticmethod
    def symbolic(g, input, conv1):
        from torch.onnx.symbolic_opset9 import _shape_as_tensor, _convolution, relu

        conv = _convolution(g, input, conv1.weight, False, 1, 1, 1, False, (), 1, None, None, None)
        output = relu(g, conv)

        return output

    @staticmethod
    def forward(self, input, conv1):
        conv = conv1(input)
        relu1 = nn.ReLU()
        res = relu1(conv)
        return res

class MyConvBlock(nn.Module):

    def __init__(self):
        super(MyConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        #self.weight = torch.tensor(self.conv1.weight, requires_grad=False)

    def forward(self, input):
        return MyConvBlockFunction.apply(input, self.conv1)

"""
In this file we try to make ONNX custom conv block op

"""
import io
import numpy as np
import torch
from torch import nn
import torch.onnx
from torch.onnx.symbolic_helper import _unimplemented, parse_args
from torch.onnx.symbolic_registry import register_op
import torch.onnx.symbolic_helper as sym_help
from torch.nn import functional as F
from torch.autograd import Function


import onnx
import onnxruntime 

OPSET_VER=10
##
## Custom ConvBlock
##

class MyConvBlockFunction(Function):

    @staticmethod
    def symbolic(g, input, conv1):
        from torch.onnx.symbolic_opset9 import _convolution, relu
        
    #     #weight =  torch.tensor(conv1.weight, requires_grad=False)
    #  #   weight = g.op("Constant", value_t=conv1.weight.clone().detach()) 
    #  #   print('_x_', weight.type(),'')
    #  #   print('_y_', conv1,'')
    #  #   print('_z_', conv1.weight,'')


    #   #  value = g.op("Constant", value_t=torch.tensor([5], requires_grad = False))

    #   #  print('_zx_', value,'')

    #     #return g.op('Relu', input)
    #     #print('\n\n ', conv1,' \n\n', conv1)
    #     print('\n\n ', g,' \n\n')

        #conv = input

        print(input)

        print('\n\n\n\n', conv1.weight.data, '\n\n\n', conv1)

        conv = _convolution(g, input, conv1.weight.data, False, 1, 1, 1, False, (), 1, None, None, None)
        output = relu(g, conv)

        return output
        

    @staticmethod
    def forward(self, input, conv1):
        conv = conv1(input)
        relu1 = nn.ReLU()
        res = relu1(conv)
        return res

class MyConvBlock(nn.Module):

    def __init__(self):
        super(MyConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels = 2, out_channels = 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        #self.weight = torch.tensor(self.conv1.weight, requires_grad=False)

    def forward(self, input):
        return MyConvBlockFunction.apply(input, self.conv1)
        #return (self.relu(self.conv1(input)))

##
## Test custom ConvBlock
##

# Make random input
N, C, H, W = 1,3,4,4
x = torch.randn(N, C, H, W, device='cpu', requires_grad=False)

# Inference
model = MyConvBlock()
model.eval()
y = model.forward(x)
# Output
print('Pytorch input', x,'\n')
print('Pytorch output', y,'\n')

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

##
## Export custom ConvBlock to ONNX
##

# Convert to onnx
# Export the model
print('\nStart conversation\n')
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "convblock.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=OPSET_VER,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])      # the model's output names

##
## Test exported onnx model
##

# Check the model
onnx_model = onnx.load('convblock.onnx')
onnx.checker.check_model(onnx_model)
print('The model is checked!')

# Inference
session = onnxruntime.InferenceSession('convblock.onnx', None)
input_name = session.get_inputs()[0].name
print('Input tensor name :', input_name)
x = x.numpy()
outputs = session.run([], {input_name: x})[0]

# Output
print('ONNX input', x.shape,'\n',x)
print('ONNX output', outputs.shape,'\n',outputs)