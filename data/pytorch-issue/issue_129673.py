from typing import Dict
import torch
import torch.nn as nn

print("torch version: ", torch.__version__)
inputs: Dict[str, torch.Tensor] = {'v0_0': torch.rand([3, 3, 3, 3])}

class Model(nn.Module):
    def forward(self, v0_0):
        # v1_0 = v0_0.unsqueeze(dim=1)
        v1_0 = torch.Tensor.unsqueeze_(v0_0, dim=1)
        return v1_0

model = Model().to(torch.device("cpu"))
for k, v in inputs.items():
    inputs[k] = v.to(torch.device("cpu"))

print('==== Eager mode ====')
ret_eager = model(**inputs)

print('==== TorchComp mode ====')
ret_exported = torch.compile(model)(**inputs)