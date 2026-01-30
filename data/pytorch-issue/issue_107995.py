import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = nn.LeakyReLU(negative_slope=5.2955089, inplace=True)

    def forward(self, inputs):
        return self.layer1(inputs)

ip_size = [0]
input_tensor = torch.randn(ip_size)
cuda_inputs = input_tensor.clone().to('cuda')

mymodel = CustomModel()
no_op_info = mymodel(input_tensor)

torch._dynamo.reset()
mymodel.to('cuda')
op_info = torch.compile(mymodel.forward, mode='reduce-overhead')(cuda_inputs)

print(op_info)