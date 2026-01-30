import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.query = torch.nn.Linear(input_size, hidden_size)
        self.key = torch.nn.Linear(input_size, hidden_size)
        self.scale_factor = torch.nn.Parameter(torch.tensor([-1.7197e+14]))

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores_scaled = scores.mul(self.scale_factor)
        t = scores_scaled.softmax(dim=-1)
        return t

input_size = 32
hidden_size = 1

x = torch.randn(10, input_size)
func = Model(input_size, hidden_size)

jit_func = torch.compile(func)

res1 = func(x) # without jit
res2 = jit_func(x)
print(res1)
# tensor([[0.], ...)
print(res2)
# tensor([[nan], ...)

-O0

nan