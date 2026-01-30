import torch.nn as nn

import torch


class NnWrapper(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, kwarg):
        result = self.op(**kwarg)
        return result


model = NnWrapper(torch.frexp)
compiled_model = torch.compile(model)

shapes = [[1, 1, 2, 3], [1, 1, 3, 3]]
for shape in shapes:
    ifm = torch.randn(shape, dtype=torch.bfloat16)
    out = [
        torch.randn(shape, dtype=torch.bfloat16),
        torch.randint(low=-50, high=50, size=shape, dtype=torch.int32),
    ]
    params = {"input": ifm, "out": out}
    res = compiled_model(params)
    print(f"res: {res}")

res: torch.return_types.frexp_out(
mantissa=tensor([[[[-0.5977, -0.6523, -0.5859],
          [-0.5117, -0.9727, -0.6914]]]], dtype=torch.bfloat16),
exponent=tensor([[[[-7,  1, -4],
          [ 0, -1, -1]]]], dtype=torch.int32))
res: torch.return_types.frexp_out(
mantissa=tensor([[[[ 0.6680, -0.8984, -0.5195],
          [-0.9219,  0.5195, -0.7148],
          [ 0.8438,  0.6523,  0.5703]]]], dtype=torch.bfloat16),
exponent=tensor([[[[-1, -2,  1],
          [ 1, -1,  0],
          [ 0, -1,  0]]]], dtype=torch.int32))