import torch
import torch.nn as nn
import copy

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = copy.copy(x)
        x = self.relu(x)
        return x

# Compilation settings
compile_setting = {'mode': None, 'fullgraph': True, 'dynamic': False}

# Input tensor
input_tensor = torch.rand([1,2]) 

# Create the model
mymodel = CustomModel()

# Forward pass
output = mymodel(input_tensor)

# Attempting to compile
op_output = torch.compile(mymodel, **compile_setting)(input_tensor)