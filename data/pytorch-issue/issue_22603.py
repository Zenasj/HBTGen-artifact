import torch

class Foo(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    def bar(self, x, dim=0):
        print(x.size(dim))

    @torch.jit.script_method
    def forward(self, x):
        self.bar(x, dim=0)

foo = Foo()