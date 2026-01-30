py
import torch
import torch.nn as nn

torch.manual_seed(420)

class Model(torch.nn.Module):
    def forward(self, x):
        x = torch.rand_like(x, device='cpu')
        return x

input_tensor = torch.randn(1, 5)

func = Model()

res1 = func(input_tensor)
print(res1)
# tensor([[0.2448, 0.8644, 0.2896, 0.1729, 0.3458]])

jit_func = torch.compile(func)
res2 = jit_func(input_tensor)
# AssertionError: While executing %rand_like_1 : [#users=1] = call_function[target=torch._inductor.overrides.rand_like](args = (%l_x_,), kwargs = {device: cpu})

x.device

device(type='cpu')

'cpu'