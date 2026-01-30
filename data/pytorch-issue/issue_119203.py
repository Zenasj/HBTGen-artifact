import torch            
import abc          
                
class A:        
    def __new__(cls, x):
        y = super().__new__(cls)
        y.y = 2 
        return y
    def __init__(self, x):
        self.x = x  
                    
@torch.compile(backend="eager")
def f(y, z):
    a = A(y)
    return a.x + a.y
        
f(torch.randn(4), torch.randn(4))