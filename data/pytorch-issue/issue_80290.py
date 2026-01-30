import torch.nn as nn

import torch

class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()      
        
    def forward(self, 
                input_: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        
        optim = torch.optim.LBFGS(tensor)
        max_iter = 50
        n_iter = 0
        
        def closure():
            optim.zero_grad()
            loss = input_ - target
            loss.backward()
            n_iter += 1
            return loss  
        
        while n_iter <= max_iter:
            optim.step(closure)
        
        return tensor
    
    
test = torch.jit.script(Test())
test.save('test.pt')