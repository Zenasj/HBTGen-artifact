import torch.nn as nn

import torch

class Module1(torch.jit.ScriptModule):
    def __init__(self):
        super(Module1, self).__init__()

    @torch.jit.script_method
    def forward(self, x, value:int):
        return x + value

@torch.jit.script
class Wrapper(object):
    def __init__(self, value):
        self.value = value

class Module2(torch.jit.ScriptModule):
    def __init__(self):
        super(Module2, self).__init__()

    @torch.jit.script_method
    def forward(self, x: Wrapper, value:int):
        return x.value + value


# from demo_class import make_module1
class Module3(torch.nn.Module):
    def __init__(self):
        super(Module3, self).__init__()
        self.module1 = Module1()
        self.module2 = Module2()

    def forward(self, x):
        #  tmp = self.module1(x,3)
        x_wrapped = Wrapper(x)
        tmp = self.module2(x_wrapped,3)
        y = tmp+1
        return y


if __name__ == "__main__":
    module = Module3()
    print(module)
    dummy = torch.zeros(1,3,200,200)
    torch.jit.trace(module, dummy)

import torch

class Module1(torch.jit.ScriptModule):
    def __init__(self):
        super(Module1, self).__init__()

    @torch.jit.script_method
    def forward(self, x, value:int):
        return x + value

@torch.jit.script
class Wrapper(object):
    def __init__(self, value):
        self.value = value

class Module2(torch.jit.ScriptModule):
    def __init__(self):
        super(Module2, self).__init__()

    @torch.jit.script_method
    def forward(self, x, value:int):
        x_wrapped = Wrapper(x)
        return x_wrapped.value + value


# from demo_class import make_module1
class Module3(torch.nn.Module):
    def __init__(self):
        super(Module3, self).__init__()
        self.module1 = Module1()
        self.module2 = Module2()

    def forward(self, x):
        #  tmp = self.module1(x,3)
        tmp = self.module2(x, 3)
        y = tmp+1
        return y


if __name__ == "__main__":
    module = Module3()
    print(module)
    dummy = torch.zeros(1,3,200,200)
    torch.jit.trace(module, dummy)