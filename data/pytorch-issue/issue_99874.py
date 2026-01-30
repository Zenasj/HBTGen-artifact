import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):
    def forward(self, x, y):
        out = x @ y
        return out

input_dim = 20
seq_length = 10
batch_size = 4

x = torch.randn(batch_size, seq_length, input_dim).cuda()
y = torch.zeros((batch_size, input_dim, seq_length)).cuda()

func = Model().to('cuda')

res1 = func(x, y)
print(res1)

jit_func = torch.compile(func)
res2 = jit_func(x, y)
# TypeError: unsupported operand type(s) for @: 'Tensor' and 'Tensor'
# While executing %matmul : [#users=1] = call_function[target=operator.matmul](args = (%l_x_, %l_y_), kwargs = {})