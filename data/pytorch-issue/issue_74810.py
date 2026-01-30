import torch.nn as nn

class SkipModule(Module):
    def __init__(self):
        super().__init__()
        self.lin = Linear(10, 10, bias=False)

    def forward(self, x):
        return self.lin(x)

class NestedLinear(Module):
    def __init__(self, fsdp_wrap):
        super().__init__()
        if fsdp_wrap:
            self.nested_linear = wrap(Linear(10, 10, bias=False).cuda())
        else:
            self.nested_linear = Linear(10, 10, bias=False).cuda()

    def forward(self, x):
        return self.nested_linear(x)


class SkipModel(Module):
    def __init__(self, double_nest):
        super().__init__()
        self.linear = nn.Linear(10, 10, bias=False).cuda()
        self.linear_skip = SkipModule().cuda()
        self.nested_linear = wrap(NestedLinear(fsdp_wrap=double_nest))

    def forward(self, x):
        x = self.linear(x)
        x = self.linear_skip(x)
        x = self.nested_linear(x)
        return x