# torch.rand(5, 5, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

    @staticmethod
    def symbolic(g, input):
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.float))
        return g.op("Clip", input, zero)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, input):
        return MyRelu.apply(input)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 5, 1, 1, dtype=torch.float32)

