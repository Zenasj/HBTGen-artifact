import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(5, 5)

    def forward(self, x):
        return self.lin(x)

mod = MyModule().cuda()
mod.forward = torch.compile(mod.forward, fullgraph=True)
input = torch.ones(1, 5).cuda()
with torch.no_grad():
    print('origin output: ', mod(input))

state_dict = mod.state_dict()
torch.save(state_dict, "test_weight.pth")
mod.lin.weight.data.fill_(0)
mod.lin.bias.data.fill_(0)
with torch.no_grad():
    print('init output: ', mod(input))

# This approach will fail 
state_dict = torch.load('test_weight.pth')
mod.load_state_dict(state_dict, assign=True)
with torch.no_grad():
    print('reload output: ', mod(input))

# This approach will success
state_dict = torch.load('test_weight.pth')
mod.lin.weight.data = state_dict['lin.weight']
mod.lin.bias.data = state_dict['lin.bias']
with torch.no_grad():
    print('reload output 2: ', mod(input))