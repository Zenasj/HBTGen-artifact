import torch
a = torch.tensor(1 ,dtype=torch.bool)                                    
b = torch.tensor(1 ,dtype=torch.bool)                                    
c = torch.tensor(0 ,dtype=torch.float16)                                 
torch.logical_and(a,b,out=c)