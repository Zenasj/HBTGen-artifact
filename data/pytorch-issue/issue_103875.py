import torch.nn as nn
import torch

import torch._inductor.config as iconfig

# iconfig.trace.enabled = True
# iconfig.trace.graph_diagram = True

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.foo = nn.Parameter(torch.rand(32, 64))
    
    def forward(self, x, y, pos_id):
        x = x.squeeze(0, 1)  # this one fails with torch.compile
        #x = x.squeeze(1).squeeze(0)  # this one goes fine with torch.compile
        x = x[pos_id].unsqueeze(1)
        res = x * y
        
        return res

model = MyModel()
model = model.to("cuda")
x = torch.rand(1, 1, 1002, 128, device="cuda")
y = torch.rand(1, 32, 1, 128, device="cuda")
pos_id = torch.Tensor([[200]]).to(torch.device("cuda"), torch.int64)

model.forward = torch.compile(model.forward, backend="eager")

with torch.no_grad():
    res = model(x, y, pos_id)

    x = torch.rand(1, 1, 543, 128, device="cuda")
    pos_id = torch.Tensor([[300]]).to(torch.device("cuda"), torch.int64)

    res = model(x, y, pos_id)