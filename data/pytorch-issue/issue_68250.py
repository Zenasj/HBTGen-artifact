import torch
import torchvision                                                       
                                                                         
m = torchvision.models.mobilenet_v3_small()                              
qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
mp = torch.quantization.quantize_fx.prepare_qat_fx(m, qconfig_dict)      
mp(torch.randn(1, 3, 224, 224))                                          
mq = torch.quantization.quantize_fx.convert_fx(mp)                       
res = mq(torch.randn(1, 3, 224, 224))