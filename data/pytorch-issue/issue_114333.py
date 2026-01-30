import torch.nn as nn
import random

import torch
torch.random.manual_seed(49993)
def fn(input):
    p = 0.2
    training = True
    inplace = False
    fn_res = torch.nn.functional.dropout2d(input, p=p, training=training, inplace=inplace, )
    return fn_res
input_tensor = torch.empty([1, 1, 32, 32], dtype=torch.float32, requires_grad=True)
torch.autograd.gradcheck(fn, input_tensor)

import torch
torch.random.manual_seed(49993)
def fn(input):
    p = 0.2
    training = True
    inplace = False
    fn_res = torch.nn.functional.dropout3d(input, p=p, training=training, inplace=inplace, )
    return fn_res
input_tensor = torch.empty([1, 1, 1, 32, 32], dtype=torch.float32, requires_grad=True)
torch.autograd.gradcheck(fn, input_tensor)

import torch
torch.random.manual_seed(49993)
def fn(input):
    p = 0.2
    training = True
    inplace = False
    fn_res = torch.nn.functional.dropout(input, p=p, training=training, inplace=inplace, )
    return fn_res

input_tensor = torch.empty([20, 16], dtype=torch.float32, requires_grad=True)
torch.autograd.gradcheck(fn, input_tensor)

def fn(input):
    arg_class = torch.nn.Dropout()(input)
    return arg_class

input_tensor = torch.tensor([16, 1], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, input_tensor)