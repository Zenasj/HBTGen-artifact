import torch.nn as nn

import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x, y_dict):
        y = y_dict['value']
        output = x + y
        return 'haha', {'output': output}


# Create the module
module = MyModule()

# Example usage
x = torch.tensor(2.0)
y = {'value': torch.tensor(3.0)}
string, output_dict = module(x, y)
print(string)  # Prints: haha
print(output_dict)  # Prints: {'output': tensor(5.)}
torch.jit.trace(module, (x,y)) # RuntimeError: Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions