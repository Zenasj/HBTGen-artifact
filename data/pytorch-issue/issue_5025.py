import torch

class OldFunction(torch.autograd.Function):
    def forward(self, *inputs):
        return [torch.FloatTensor([1])]

class NewFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        return [torch.FloatTensor([1])]

OldFunction()()
# RuntimeError: _Map_base::at

# NewFunction.apply()
# RuntimeError: _Map_base::at