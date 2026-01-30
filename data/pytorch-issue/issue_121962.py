import torch.nn as nn

import torch
import torch.nn.functional as F

torch.set_printoptions(20)
inputs=torch.randn(2,6, dtype=torch.float16)
weights=torch.randn(2,6, dtype=torch.float16)

outputs=F.linear(
    inputs,
    inputs
)

input1, input2=inputs.split(3,dim=1)
weight1, weight2=inputs.split(3, dim=1)

dist_output=F.linear(input1, weight1)+F.linear(input2, weight2)
print(
    dist_output==outputs,'\n',
    outputs,'\n',
    dist_output, '\n'
)