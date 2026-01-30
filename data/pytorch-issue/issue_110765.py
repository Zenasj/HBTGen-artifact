import torch
import torch.nn as nn

_dummy_compiler_variable = 1 # also works with _dummy_compiler_variable = torch.tensor([1])

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_pre_hook(self.pre_forward, with_kwargs=True)

    def pre_forward(self, module, args, kwargs):
        if torch._utils.is_compiling():
            global _dummy_compiler_variable
            _dummy_compiler_variable += 1
            print("path A")   
            return args, kwargs

        print("path B")   
        return args, kwargs

    def forward(self, x):
        return x

mod = MyModule()
m = torch.compile(mod, backend="eager")
m(torch.tensor([1])) # prints "path A"

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_pre_hook(self.pre_forward, with_kwargs=True)
        self.val = 0 # if this is 0, we trigger recompilation. If this is torch.tensor([1]), we do not
        # I think this is expected as the compiled code is guarded on the value of `val`

    def pre_forward(self, module, args, kwargs):
        if torch._utils.is_compiling():
            self.val += 1
            # print("A")
            return args, kwargs

        print("path B")   
        return args, kwargs

    def forward(self, x):
        return x

mod = MyModule()
m = torch.compile(mod, backend="eager")

print("COMPILING\n")
m(torch.tensor([1]))
print(_dummy_compiler_variable)

print("VAL", m.val) # 0
# 1 if we uncomment print("A"), triggering writing to graph

m(torch.tensor([1]))
print("VAL", m.val) # 0 
# 2 if we uncomment print("A"), triggering writing to graph