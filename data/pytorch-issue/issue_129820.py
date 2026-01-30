import torch
import torch.nn as nn
print("torch version: ",torch.__version__)

inputs = {'v3_0': torch.randn(7, 7), 'v1_0': torch.randn(()), 'v0_0': torch.randn(7, 7)}
class Model(nn.Module):
    def forward(self, v3_0, v1_0, v0_0):
        # v3_0: [7, 7], torch.float32
        # v1_0: [], torch.float32
        # v0_0: [7, 7], torch.float32
        v4_0 = torch.lt(v1_0, other=40, out=v3_0)
        print(v1_0.shape, "==>", v4_0.shape)
        return v4_0

model = Model().to(torch.device("cpu"))

print('==== TorchComp mode ====')
ret_exported = torch.compile(model)(**inputs)
print("returned value's shape :", ret_exported.shape)