import torch

code = '''
def forward(self, x):
    return torch.neg(x,)
'''

invoke = '''
print(forward(None, torch.rand(3, 4)))
'''

exec(code + invoke, {'torch': torch}, {})

class SM(torch.jit.ScriptModule):
    def __init__(self):
        super(SM, self).__init__()
        self.define(code)

sm = SM()
print(sm(torch.rand(3, 4)))