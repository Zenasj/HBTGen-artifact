import torch.nn as nn

import torch                                                                                    
import torch.nn.functional as F                                         
import torch.ao.quantization.quantize_fx as quantize_fx                                                        
                                                                        
class M(torch.nn.Module):                                               
    def __init__(self):                                                 
        super().__init__()                                              
        self.w = torch.nn.Parameter(torch.randn(1, 1))                  
        self.b = torch.nn.Parameter(torch.randn(1))                     
                                                                        
    def forward(self, x):                                               
        x = F.linear(x, self.w, self.b)                                 
        return x                                                        
                                                                        
m = M().eval()                                                          
mp = quantize_fx.prepare_fx(m, {'': torch.quantization.default_qconfig})
mq = quantize_fx.convert_fx(mp)