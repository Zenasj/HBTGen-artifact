import torch.nn as nn

import torch
from torch import nn

class MyDropout(nn.Module):
    """Simple module that only includes a Dropout layer"""
    def __init__(self):
        super(MyDropout, self).__init__()
        self.dr = nn.Dropout(p=0.5)
    
    def forward(self, data):
        return self.dr(data)

# Module instantiation
my_drop = MyDropout()
my_drop.eval()

# Create ScriptModule
my_drop_scripted = torch.jit.script(my_drop)
my_drop_scripted.eval()

# Test if forward pass works:
input_data = torch.Tensor([[1,2,3], [4,5,6]])
output_data = my_drop_scripted(input_data)
print(output_data)

# Export to ONNX
torch.onnx.export(my_drop_scripted,
                  input_data,
                  "dropout_model.onnx",
                  verbose=True,
                  example_outputs=output_data)