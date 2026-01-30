import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self, dropout_p):
        super(Model, self).__init__()
        self.dropout_p = dropout_p

    def forward(self, x):
        out = torch.nn.functional.dropout(x, p=self.dropout_p)
        return out

input_dim = 4
batch_size = 3
dropout_p = 1

x = torch.randn(batch_size, input_dim)
func = Model(dropout_p).to('cpu')

res1 = func(x)
print(res1)

jit_func = torch.compile(func)
res2 = jit_func(x)
# ZeroDivisionError: float division by zero
# While executing %lowmem_dropout : [#users=1] = call_function[target=torch._inductor.overrides.lowmem_dropout](args = (%l_x_,), kwargs = {p: 1})

py
import torch

torch.manual_seed(420)

class MyModel(torch.nn.Module):

    def __init__(self, dropout_p):
        super(MyModel, self).__init__()
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(self.dropout_p)

    def forward(self, x):
        out = self.dropout(x)
        return out

input_dim = 5
dropout_p = 1
batch_size = 3

x = torch.rand(batch_size, input_dim)
func = MyModel(dropout_p).to('cpu')
test_inputs = [x]

res1 = func(x)
print(res1)
# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
# tensor([[0.8054, 0.1990, 0.9759, 0.1028, 0.3475],
#         [0.1554, 0.8856, 0.6876, 0.2506, 0.1133],
#         [0.2105, 0.4035, 0.2448, 0.8644, 0.2896]])

import torch

torch.manual_seed(420)

class MyModel(torch.nn.Module):

    def __init__(self, dropout_p):
        super(MyModel, self).__init__()
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(self.dropout_p)

    def forward(self, x):
        out = self.dropout(x)
        return out

input_dim = 5
dropout_p = 1
batch_size = 3

x = torch.rand(batch_size, input_dim)
func = MyModel(dropout_p).to('cpu')
test_inputs = [x]

with torch.no_grad():
    func.train(False)
    res1 = func(x)
    print(res1)
# tensor([[0.8054, 0.1990, 0.9759, 0.1028, 0.3475],
#         [0.1554, 0.8856, 0.6876, 0.2506, 0.1133],
#         [0.2105, 0.4035, 0.2448, 0.8644, 0.2896]])

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
# tensor([[0.8054, 0.1990, 0.9759, 0.1028, 0.3475],
#         [0.1554, 0.8856, 0.6876, 0.2506, 0.1133],
#         [0.2105, 0.4035, 0.2448, 0.8644, 0.2896]])