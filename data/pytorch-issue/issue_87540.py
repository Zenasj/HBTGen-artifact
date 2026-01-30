import torch
import torch.nn as nn

def print_hook(module: nn.Module, _inputs, _outputs):
    print('hook triggered on', module)

class TestModule(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.linear = nn.Linear(s, s)

    def forward(self, x):
        x = self.linear(x)
        return {'x': x}
        #return [x, x]
        #return x, x

x = torch.ones([10, 10])
model = TestModule(10)
model.register_full_backward_hook(print_hook)
y = model(x)['x']
loss = (y**2).sum()

print('backward started')
loss.backward()
print('backward complete')