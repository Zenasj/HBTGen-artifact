import torch.nn as nn

py
import torch

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        t1 = torch.full([2, 4], 1)
        t2 = t1.to(dtype=torch.bool)
        # t2 = torch.full([2, 4], 1, dtype=torch.bool) # if use this line, the output with compile with be [1, 1, 1, 1]
        t3 = torch.cumsum(t2, 1)
        return t3

with torch.no_grad():
    func = Model()
    func = torch.compile(func)
    print(func())
    # tensor([[True, True, True, True],
    #         [True, True, True, True]])

func = Model()
print(func())
# tensor([[1, 2, 3, 4],
#         [1, 2, 3, 4]])