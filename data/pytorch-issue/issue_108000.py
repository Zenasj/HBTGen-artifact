import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

    def forward(self, inputs):
        return torch.logical_not(**inputs)

ip_size = [1, 0]
input_tensor = torch.randn(ip_size)
cuda_inputs = input_tensor.clone().to('cuda')
out = torch.empty(0)
cuda_out = input_tensor.clone().to('cuda')

mymodel = CustomModel()
no_op_info = mymodel({'input': input_tensor, 'out': out})

mymodel.to('cuda')
op_info = torch.compile(mymodel.forward, mode='reduce-overhead')({'input': cuda_inputs, 'out': cuda_out})

print(op_info)