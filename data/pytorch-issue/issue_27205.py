python
import torch
import torch.nn as nn

class testModel(nn.Module):
    def __init__(self):
        super(testModel,self).__init__()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        x = self.dropout(x)
        return x
    
model = testModel()
x = torch.rand(10, dtype = torch.float32)
# Dropout is a no-op for eval
# This works
print(x)
model.eval()
print(model(x))

xq = torch.quantize_per_tensor(x, 0.1 , 0, torch.quint8)
# Runs with quantized tensors
print(xq)
print(model(xq))
scriptmodel = torch.jit.script(model)
# Scriptmodel does not run with quantized tensor
scriptmodel(xq)

import torch
import torch.nn as nn

class testModel(nn.Module):
    def __init__(self):
        super(testModel,self).__init__()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        x = self.dropout(x)
        return x
    
model = testModel()
x = torch.rand(10, dtype = torch.float32)
# Dropout is a no-op for eval
# This works
print(x)
model.eval()
print(model(x))

xq = torch.quantize_per_tensor(x, 0.1 , 0, torch.quint8)
# Runs with quantized tensors
print(xq)
print(model(xq))
scriptmodel = torch.jit.script(model)
scriptmodel.eval()
# Scriptmodel does not run with quantized tensor
scriptmodel(xq)