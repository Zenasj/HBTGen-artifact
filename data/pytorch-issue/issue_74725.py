import torch
from torch.autograd import Function
from numpy import flip
import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.autograd import Variable

class ScipyConv2dFunction(Function):
  @staticmethod
  def forward(ctx, input, filter, bias, params0, params1):
    # detach so we can cast to NumPy
    input, filter, bias = input.detach(), filter.detach(), bias.detach()
    # Creating new lists from the list of params in a basic way. We add 0.01
    # to each value in the list to represent a change between the input
    # list and the output list. In practice the values of the list may change
    # much more depending on the functionality of the forward pass. For purposes
    # of brevity we keep such a change to be only "+ 0.01"
    param_new0=[]
    for param in params0:
      param_new0.append(param.detach().numpy()[0] + 0.01)
    param_new1=[]
    for param in params1:
      param_new1.append(param.detach().numpy()[0] + 0.01)

    result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
    result += bias.numpy()

    param_new0 = torch.as_tensor(param_new0).to(torch.float64)
    param_new1 = torch.as_tensor(param_new1).to(torch.float64)
    ctx.save_for_backward(input, filter, bias, param_new0, param_new1)
    return torch.as_tensor(result, dtype=input.dtype)

  @staticmethod
  def backward(ctx, grad_output):
    grad_output = grad_output.detach()
    input, filter, bias, param_new0, param_new1 = ctx.saved_tensors
    grad_output = grad_output.numpy()
    # Converting the parameter lists to NumPy
    param_new0 = param_new0.numpy()
    param_new1 = param_new1.numpy()
    # Making a small change in the lists to represent a change due to backprop.
    # In practice this change may be much larger, however for purposes of
    # brevity we keep such a change to be only "- 0.05"
    param_new0 = param_new0 - 0.05
    param_new1 = param_new1 - 0.05
    grad_bias = np.sum(grad_output, keepdims=True)
    grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
    # the previous line can be expressed equivalently as:
    # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
    grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
    # returning our gradients including our parameter lists
    return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float), torch.from_numpy(param_new0).to(torch.float), torch.from_numpy(param_new1).to(torch.float) 

class ScipyConv2d(Module):
  def __init__(self, filter_width, filter_height):
    super(ScipyConv2d, self).__init__()
    self.filter = Parameter(torch.randn(filter_width, filter_height))
    self.bias = Parameter(torch.randn(1, 1))
    # Defining new ParameterLists
    self.params0 = nn.ParameterList([Parameter(torch.randn(1)), Parameter(torch.randn(1))])
    self.params1 = nn.ParameterList([Parameter(torch.randn(1)), Parameter(torch.randn(1))])

  def forward(self, input):
    return ScipyConv2dFunction.apply(input, self.filter, self.bias, self.params0, self.params1)
    #return ScipyConv2dFunction.apply(input, self.filter, self.bias, Variable(self.params0), Variable(self.params1))

module = ScipyConv2d(3, 3)
print("Filter and bias: ", list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print("Output from the convolution: ", output)
output.backward(torch.randn(8, 8))
print("Gradient for the input map: ", input.grad)