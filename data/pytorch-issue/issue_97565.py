import torch
import torch.nn as nn

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        print("forward")
        return self.relu(self.linear(x))
    
    def compile(self, *args, **kwargs):
        print("compile")
        self.__call__ = torch.compile(self, *args, **kwargs) # torch.compile(self/self.forward/self._call_impl,self.__call__) all behave the same way

m = ToyModel()
m.compile()
m(torch.randn(10, 10))