import torch                                                    
                                                                
class exampleFct(torch.autograd.Function):                      
    @staticmethod                                               
    def forward(self, x):                                       
        self.save_for_backward(x)                               
        return x ** 2                                           
                                                                
    @staticmethod                                               
    def backward(self, dy):                                     
        x, = self.saved_variables                               
        with torch.enable_grad():                               
            y = x ** 2                                          
            return torch.autograd.grad(y, x, dy)
                                                                
                                                                
x = torch.tensor([[3, 4]], requires_grad=True)                  
m = exampleFct.apply(x).sum().backward()                        
print(x.grad.data)

import torch                                                    
                                                                
class exampleFct(torch.autograd.Function):                      
    @staticmethod                                               
    def forward(self, x):                                       
        y = x ** 2  
        self.save_for_backward(x, y)  
        return y                             
                                         
                                                                
    @staticmethod                                               
    def backward(self, dy):                                     
        x, y = self.saved_tensors                               
        with torch.enable_grad():                               
            return torch.autograd.grad(y, x, dy)
                                                                
                                                                
x = torch.tensor([[3, 4]], dtype=torch.float, requires_grad=True)                  
m = exampleFct.apply(x).sum().backward()                        
print(x.grad.data)