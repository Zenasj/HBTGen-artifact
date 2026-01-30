import torch

a=torch.randn(3, dtype=torch.complex64)                                                                                                                                                                
b=torch.randn(3, dtype=torch.complex64)                                                                                                                                                                
c=torch.randn(3,3, dtype=torch.complex64)   
c.addr(a,b, beta=0+0.1j)