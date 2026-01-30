import torch.nn as nn

import torch
from enum import Enum

class MyEnum(Enum):
    A = "a"
    B = "b"
    C = "c"

class SomeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.my_enum = MyEnum.A
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x) -> torch.Tensor:
        if self.my_enum not in {MyEnum.A, MyEnum.B}:
            return x
        return self.linear(x)

x = torch.randn(1, 1)
model = SomeModel()
explain = torch._dynamo.explain(model)(x)
print(explain)