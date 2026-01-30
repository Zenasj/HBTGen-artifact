import torch                                                           
import torch.nn as nn                                                  
                                                                       
m = nn.Sequential(                                                     
    nn.Conv2d(1, 1, 1),                                                
)                                                                      
m.eval()                                                               
                                                                       
qconfig_dict = {'': torch.quantization.get_default_qconfig('qnnpack')} 
mp = torch.quantization.prepare_fx(m, qconfig_dict)