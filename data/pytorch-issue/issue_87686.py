import copy                                                          
import torch                                                         
import torch.nn as nn                                                
import torch.fx                                                      
import torch.nn.functional as F                                      
import torch.ao.quantization.quantize_fx as quantize_fx              
                                                                     
import torch                                                         
from torch.ao.quantization import get_default_qconfig_mapping        
from torch.quantization.quantize_fx import prepare_fx, convert_fx    
import copy                                                          
                                                                     
class M(torch.nn.Module):                                            
    def __init__(self):                                              
        super().__init__()                                           
        self.w = torch.nn.Parameter(torch.randn(1, 1))               
                                                                     
    def forward(self, x):                                            
        x = F.linear(input=x, weight=self.w)                         
        return x                                                     
                                                                     
m = M()                                                              
mp = quantize_fx.prepare_fx(                                         
    m, get_default_qconfig_mapping('fbgemm'), (torch.randn(1, 1),))  
mq = quantize_fx.convert_fx(mp)