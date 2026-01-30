import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantize_fx as quantize_fx

class M(nn.Module):                                               
    def __init__(self):                                           
        super().__init__()                                        
        self.w = torch.randn(1, 1)                                
        self.b = torch.randn(1)                                   
                                                                  
    def forward(self, x):                                         
        x = F.linear(x, weight=self.w, bias=self.b)               
        return x                                                  
                                                                  
m = M().eval()                                                    
qconfig_dict = {'': torch.ao.quantization.default_qconfig}        
mp = quantize_fx.prepare_fx(m, qconfig_dict, (torch.randn(1, 1),))
mq = quantize_fx.convert_fx(mp)                                   
print(mq)