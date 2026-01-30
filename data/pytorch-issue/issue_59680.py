import torch

def a(x):
    return torch.neg(torch.relu(torch.relu(x)))

def b(x):
    return torch.relu(torch.neg(torch.relu(x)))

import re

class DebugNamespace(torch.fx.graph._Namespace):
    def __init__(self):
        super().__init__()
        self.re_pattern = re.compile(r'_\d+$')

    def create_name(self, candidate: str, obj: Optional[Any]) -> str:
        return self.re_pattern.sub('', candidate)

traced_a = torch.fx.symbolic_trace(a).graph.stripped_python_code('self').splitlines(keepends=True)
traced_b = torch.fx.symbolic_trace(b).graph.stripped_python_code('self').splitlines(keepends=True)

import difflib

diff_str = '\n'.join(difflib.ndiff(traced_a, traced_b))
print(diff_str)