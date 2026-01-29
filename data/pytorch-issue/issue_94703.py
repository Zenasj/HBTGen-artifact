# torch.rand(1, 3, 4, 4, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
from torch.autograd import Function

# Custom op for ONNX representation
class MyStrangeOp(Function):
    @staticmethod
    def symbolic(g, input, weight, bias, floatAttr, intAttr):
        return g.op("MyStrangeOp", input, weight, bias, float_attr_f=floatAttr, int_attr_i=intAttr), \
               g.op("MyStrangeOp", input, weight, bias, float_attr_f=floatAttr, int_attr_i=intAttr)

    @staticmethod
    def forward(ctx, input, weight, bias, floatAttr, intAttr):
        return input + weight, input * weight + bias

myStrangeOpForward = MyStrangeOp.apply

# Layer using the custom op
class MyStrangeOpLayer(nn.Module):
    def __init__(self, weight, bias, floatAttr, intAttr):
        super(MyStrangeOpLayer, self).__init__()
        self.weight = weight
        self.bias = bias
        self.floatAttr = floatAttr
        self.intAttr = intAttr

    def forward(self, x):
        return myStrangeOpForward(x, self.weight, self.bias, self.floatAttr, self.intAttr)

# Model using the custom layer
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.myLayer1 = MyStrangeOpLayer(weight=nn.Parameter(torch.ones(1, 3, 4, 4)), 
                                         bias=nn.Parameter(torch.ones(1, 3, 4, 4)), 
                                         floatAttr=[0.1, 0.5], 
                                         intAttr=[2, 2])
        self.myLayer2 = MyStrangeOpLayer(weight=nn.Parameter(torch.ones(1, 3, 4, 4)), 
                                         bias=nn.Parameter(torch.ones(1, 3, 4, 4)), 
                                         floatAttr=[0.5, 0.5], 
                                         intAttr=[3, 3])
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        x1, x2 = self.myLayer1(x)
        x3, x4 = self.myLayer2(x)
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x3 = self.conv1(x3)
        x4 = self.conv1(x4)
        return x1 + x2 + x3 + x4

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 4, 4, dtype=torch.float32)

