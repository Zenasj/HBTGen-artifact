import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(2))
        self.linear = nn.Linear(2, 2)
        self.attr = torch.randn(2)
        self.attr2 = torch.randn(2)

    def forward(self, x):
        return self.linear(self.W + (self.attr + self.attr2) + x)

mod = fx.symbolic_trace(Test())
mod.to_folder('foo', 'Foo')

import torch
class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        state_dict = torch.load('foo/state_dict.pt')
        self.linear = torch.load('foo/linear.pt') # Linear(in_features=2, out_features=2, bias=True)
        self.__tensor_constant0 = state_dict['__tensor_constant0']
        self.W = torch.nn.Parameter(state_dict['W'])

    def forward(self, x):
        w = self.W
        tensor_constant0 = self.__tensor_constant0
        add_1 = w + tensor_constant0
        add_2 = add_1 + x
        linear_1 = self.linear(add_2)
        return linear_1