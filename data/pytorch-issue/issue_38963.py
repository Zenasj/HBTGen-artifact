import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
        self._ml_len: int = len(self.linears)
    def forward(self, x: float = 0):
        m = self._ml_len
        return m

model = MyModule()
print(model())
m_script = torch.jit.script(model)