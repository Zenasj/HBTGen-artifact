import torch.nn as nn
import random

import torch
from torch.autograd import forward_ad
torch.random.manual_seed(49993)
def fn(input):
    inplace = False
    fn_res = torch.nn.functional.relu(input, inplace=inplace )
    return fn_res

input_tensor = torch.empty([1, 1, 16, 16], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, input_tensor)

import torch
from torch.autograd import forward_ad
torch.random.manual_seed(49993)
def fn(input):
    negative_slope = 0.2
    inplace = False
    fn_res = torch.nn.functional.leaky_relu(input, negative_slope=negative_slope, inplace=inplace, )
    return fn_res
input_tensor = torch.empty([1, 1, 7, 7], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, input_tensor)

import torch
from torch.autograd import forward_ad
torch.random.manual_seed(49993)
def fn(input, weight):
    fn_res = torch.nn.functional.prelu(input, weight, )
    return fn_res
input_tensor = torch.empty([2], dtype=torch.float64, requires_grad=True)
weight_tensor = torch.empty([1], dtype=torch.float64, requires_grad=True)
input_tensor_list = [input_tensor, weight_tensor]
torch.autograd.gradcheck(fn, input_tensor_list)