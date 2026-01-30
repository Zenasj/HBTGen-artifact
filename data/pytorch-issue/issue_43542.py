import torch.nn as nn
 
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.foo
 
    @property
    def foo(self):
        return [].bar  # This error cannot be traced correctly
        # return error  # This error can be traced correctly
 
Module()

class MyClass:

    @property
    def foo_error(self):
        return [].bar
        # raise AttributeError # or raise the error directly (it gets suppressed)
    
    def __getattr__(self, name):
        return object.__getattr__(name)

print(MyClass().foo_error)