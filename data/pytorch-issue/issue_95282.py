import torch.nn as nn

import torch
import torch._dynamo
import logging
torch._dynamo.config.log_level = logging.INFO 
torch._dynamo.config.output_code = True

class MyModule(torch.nn.Module):
    def __init__(self, flag):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2)) 
        self.flag= flag

    def forward(self, x): 
        if self.flag is True:
            x = self.conv(x)
        else:
            x = torch.cat((x, x)) 
        return x

def my_compiler(gm, example_inputs):
    #  Have some gm transformation
    return gm.forward

model = MyModule(True)
opt_model = torch._dynamo.optimize(my_compiler)(model)

input = torch.randn(20, 16, 50, 100)
opt_model.eval()
_ = opt_model(input)
# need to capture the transformed script graph
script_model = torch.jit.trace(opt_model, input)