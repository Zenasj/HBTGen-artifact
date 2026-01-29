# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyScriptModule(torch.nn.Module):
    @torch.jit.export
    def forward(self):
        a: Dict[str, int] = {}
        for i in range(3):
            a.update({'a': i})
        return torch.tensor(a['a'], dtype=torch.int64)

class MyPythonModule(nn.Module):
    def forward(self):
        a = {}
        for i in range(3):
            a.update({'a': i})
        return torch.tensor(a['a'], dtype=torch.int64)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scripted = torch.jit.script(MyScriptModule())
        self.python = MyPythonModule()

    def forward(self, x):
        script_out = self.scripted()
        python_out = self.python()
        return script_out != python_out  # Returns True if outputs differ

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

