import torch.nn as nn

class ModuleWithOptionallyMissingAttributes(nn.Module):
    def __init__(self, a: Optional[nn.Module] = None):
        super().__init__()
        self.has_a = a is not None
        self.a = a or nn.Identity()
        if a is not None:
            self.b = nn.Linear(2, 2)

    def forward(self, input):
        if not self.has_a:
            return input
        if hasattr(self, "b"):
            return self.a(self.b(input))

jit.script(ModuleWithOptionallyMissingAttributes(nn.Linear(2, 3)))
jit.script(ModuleWithOptionallyMissingAttributes())