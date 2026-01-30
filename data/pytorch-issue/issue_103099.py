import torch.nn as nn

import torch
torch._dynamo.config.verbose=True

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x, step):
        return {f'output_{step}': self.lin(x[f'input_{step}'])}

mod = MyModule()
opt_mod = torch.compile(mod)

my_input = {
    'input_0': torch.ones([100]),
    'input_1': torch.ones([100])
}

for step in range(2):
    output = opt_mod(my_input, step)
    print(output.keys())