import torch.nn as nn

import torch
from torch import nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

    def forward(self, inputs):
        return torch.cosh(**inputs)

ip_size = [1]
input_tensor = torch.randn(ip_size)
out_tensor = torch.empty([2])
cuda_inputs = input_tensor.clone().to('cuda')
cuda_out = out_tensor.clone().to('cuda')

mymodel = CustomModel()
no_op_info= mymodel({'input': input_tensor, 'out': out_tensor})
mymodel.to('cuda')
op_info = torch.compile(mymodel)({'input': cuda_inputs, 'out': cuda_out})
print(op_info)