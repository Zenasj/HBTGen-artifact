import torch
import torch.nn as nn

from torch._inductor import config
config.fallback_random = True
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, other):
        return torch.bitwise_right_shift(input=input, other=other)


input = torch.tensor(1000, dtype=torch.int64)
other = torch.tensor(64, dtype=torch.int64)

inputs = [input, other]

model = Model()
output = model(*inputs)

c_m = torch.compile(model)
c_output = c_m(*inputs)

print(output)
print(c_output)