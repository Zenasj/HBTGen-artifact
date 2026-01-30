import torch.nn as nn

import torch

class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo, self).__init__()
        self.bn = torch.nn.BatchNorm1d(3)
    
    def forward(self, x):
        return self.bn(x)

def main():
    x1 = torch.ones(1,3)
    x2 = torch.ones(2,3)
    model = Foo()
    y = model(x1) # this line will fail to run 
    y = model(x2) # this line can run without error

if __name__ == '__main__':
    main()