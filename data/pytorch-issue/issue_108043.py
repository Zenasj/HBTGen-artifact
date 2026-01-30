import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = nn.ConstantPad2d(padding=0, value=1)
        
    def forward(self, inputs):
        return self.layer1(inputs)

ip_size = [0, 0, 1]
input_tensor = torch.randn(ip_size)
cuda_inputs = input_tensor.clone().to('cuda')

mymodel = CustomModel()
no_op_info = mymodel(input_tensor)
mymodel.to('cuda')
op_info = torch.compile(mymodel.forward, mode='max-autotune')(cuda_inputs)

import torch._inductor.config
torch._inductor.config.triton.cudagraph_trees = False