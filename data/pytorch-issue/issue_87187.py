import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100,100)
    
    def forward(self,x):
        return self.embed(x)
    
model = Model()

model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model = torch.quantization.prepare_qat(model)

torch.quantization.convert(model)

model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model = torch.quantization.prepare_qat(model)

propagate_qconfig_(model, qconfig_dict=None) ### In torch.ao.quantization.prepare_qat