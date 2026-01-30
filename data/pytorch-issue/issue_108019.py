import torch.nn as nn

import torch

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
    def forward(self, inputs):
        return torch.nn.SELU(inplace=True)(inputs)

input_tensor = torch.empty(0)
cuda_inputs = input_tensor.clone().to('cuda')

mymodel = CustomModel()
# Create the model
no_op_info= mymodel(input_tensor)
print(no_op_info)

# torch._dynamo.reset()
mymodel.to('cuda')
op_info = torch.compile(mymodel)(cuda_inputs)
print(no_op_info)