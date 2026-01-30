import torch, torch_xla                                                                                                                                                                                                                                     
import torch_xla.core.xla_model as xm                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                            
device = xm.xla_device()                                                                                                                                                                                                                                    
t1 = torch.tensor(10, device=device)                                                                                                                                                                                                                        
print(torch.div(t1, 2.0))

import torch, torch_xla                                                                                                                                                                                                         
import torch_xla.core.xla_model as xm                                                                                                                                                                                           
                                                                                                                                                                                                                                
device = xm.xla_device()                                                                                                                                                                                                        
t1 = torch.tensor(10, device=device)                                                                                                                                                                                            
print(torch.div(t1, 2.0))

torch.div(t1, 2.0)