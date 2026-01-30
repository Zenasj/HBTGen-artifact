import torch

def foo(x):                       
    x = x.to(torch.float8_e4m3fn) 
    return x                      
                                  
foo = torch.compile(foo)          
x = torch.randn(2, 2)             
x = foo(x)                        
print(x)