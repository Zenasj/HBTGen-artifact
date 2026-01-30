import torch.nn as nn

import torch
m = torch.nn.AdaptiveAvgPool3d((1,1,1))

inputs = torch.rand([8,54,16,56,56])
inputs = inputs.cuda()
inputs_2 = inputs.half()

out = m(inputs)
out2 = m(inputs_2)

print('Discepancies', torch.sum(torch.abs(out2- out)) )

import torch
data = torch.load('example_data.pt')
m = torch.nn.AdaptiveAvgPool3d((1,1,1))
with torch.cuda.amp.autocast():
    out = m(data)
print(torch.max(out))