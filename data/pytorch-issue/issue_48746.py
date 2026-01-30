import torch.nn as nn

import torch
from typing import Optional

def print_params_and_buffers(main_model):
    print("---parameters---")
    for name, param in main_model.named_parameters():
        print(f"name {name}")

    print("----buffers---")
    for name, buffer in main_model.named_buffers():
        print(f"name {name}")

class Net(torch.nn.Module):

    # removing the type annotation keeps the buffer_param after scripting
    buffer_param: Optional[torch.Tensor]
    param: Optional[torch.Tensor]

    def __init__(self):
        super(Net, self).__init__()
        self.register_buffer("buffer_param", torch.zeros(2,2))
        self.param = torch.nn.Parameter(torch.ones(2,2))

    def forward(self, x):
        return x

net = Net()

print("====eager module====")
print_params_and_buffers(net)

print("\n====scripted module====")
scripted_net = torch.jit.script(net)
print_params_and_buffers(scripted_net)