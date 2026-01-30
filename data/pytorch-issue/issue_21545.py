import torch
from typing import List

class Foo(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, languages: List[str]=[], per_token_languages: List[List[str]]=[]):
        return torch.rand(3, 4)

f = Foo()
f.save('/tmp/test.zip')

torch.jit.load('/tmp/test.zip')

def test(x=[]):
   x.append(1)
   return len(x)

print(test()) # 1
print(test()) # 2