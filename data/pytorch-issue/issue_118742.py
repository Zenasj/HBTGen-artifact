python
import torch
import torch.nn as nn
print("torch version: ",torch.__version__)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin  = nn.Linear(10, 30)
        self.lin  = nn.utils.weight_norm(self.lin,dim=0)
    def forward(self, inp1):
        print("inp1 size :", inp1.size())
        out = self.lin(inp1)
        return out
inputs = [ torch.randn(10) ]
model = Model().to(torch.device("cpu"))
print('==== Eager mode ====')
ret_eager = model(*inputs)

print('==== TorchComp mode ====')
ret_exported = torch.compile(model)(*inputs)
print('OK!')