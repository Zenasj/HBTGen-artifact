import torch
import torch.nn as nn
import torch.nn.functional as F

dummy_data = torch.randn(1,  784)
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(784, 256)
        self.output = torch.nn.Linear(256, 10)
        
    def forward(self, x):
        mask = torch.zeros((1,392),dtype=torch.bool)
        mask = F.pad(mask, (0, 392), value=True)
        x.masked_fill_(mask, .314)
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x
      
m = Network()
torch.onnx.export(m,                
                  dummy_data,       
                  "smoke_test.onnx",
                  export_params=True,        
                  opset_version=14,          
                  do_constant_folding=True,  
                  input_names  = ['input_ids'],   
                  output_names = ['logits'],  
                  verbose=True
                 )