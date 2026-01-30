import copy
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(in_features=1, out_features=2)
    
    def forward(self, x):
        y = self.layer0(x)
        return y

with torch.no_grad():
    model = Model()
    i0 = torch.rand(1) # leads to ERROR 2
    # i0 = torch.rand(2, 1) # NOTE: this shape works fine
    model(i0)

model_tmp = copy.deepcopy(model)
compiled = torch.compile(model_tmp)
compiled(i0)
print(f'==== here ====')

with torch.no_grad():
    # model.eval() # NOTE: report ERROR 1 w/o this line
    compiled = torch.compile(model)
    compiled(i0)