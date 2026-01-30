import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.branch1 = nn.Sequential()
    
    def forward(self, x):
        return x
    
model = Test()
model.eval()
model.qconfig = torch.quantization.get_default_qconfig()
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)
print(model)
torch.jit.script(model)