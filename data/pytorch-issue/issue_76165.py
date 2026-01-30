import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.c = []
        self.init_weight()
        names = self.__dict__
        for i in range(len(self.c)):
            names['weight_' + str(i)] = Parameter(self.c[i], requires_grad=True)
    
    def init_weight(self):
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 3, requires_grad=True)
        self.c.append(a); self.c.append(b)
    
    def forward(self):
        return self.a + self.b

test = Test()
for param in test.parameters():
    print(param)

import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.c = []
        self.init_weight()
        for i in range(len(self.c)):
            setattr(self, 'weight_' + str(i), nn.Parameter(self.c[i], requires_grad=True))
    
    def init_weight(self):
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 3, requires_grad=True)
        self.c.append(a); self.c.append(b)
    
    def forward(self):
        return self.a + self.b

test = Test()
for param in test.parameters():
    print(param)