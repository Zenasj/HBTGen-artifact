import torch.nn as nn

import torch

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        return run_function(*args)

def checkpoint(function, *args):
    return CheckpointFunction.apply(function, *args)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
  
    def forward(self, input):
        def custom_forward(*inputs):
            x_ = inputs[0]
            return x_

        return checkpoint(custom_forward, input)

  
model = Net()
input = torch.zeros(2)
output = model(input, )
print(output)

with torch.no_grad():
    torch.onnx.export(model, (input, ), './check.onnx',)