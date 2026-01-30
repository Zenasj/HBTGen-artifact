import torch.nn as nn

py
import torch

torch.manual_seed(420)

x = torch.randn(1, 10)

class Model(torch.nn.Module):

    def __init__(self, in_features=10, out_features=5):
        super(Model, self).__init__()
        self.linear = torch.nn.quantized.dynamic.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.linear(x)
        return x

func = Model().to('cpu')


with torch.no_grad():
    func.train(False)
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # AttributeError: __torch__.torch.classes.quantized.LinearPackedParamsBase (of Python compilation unit at: 0) does not have a field with name 'is_complex'