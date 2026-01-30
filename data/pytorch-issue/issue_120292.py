import torch
import torch.nn as nn
print("torch version: ",torch.__version__)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = torch.rand([9],dtype=torch.float32) # [9], torch.float32


    def forward(self):
        # v0_0: [], torch.int32
        v4_0 = torch.nn.functional.rrelu(input=self.p0, training=True)
        return v4_0

inputs = {}
model = Model().to(torch.device("cpu"))
ret_exported = torch.compile(model)(**inputs)