import torch.nn as nn

import torch
from typing import Any, List

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
    
    def call(self, input1: str, input2: str) -> str:
        return input1

    def forward(self, input: Any) -> None:
        if torch.jit.is_scripting() and isinstance(input, List[str]):
            # This can be a big code block in actual use cases
            for el in input:
                print(el)
        elif not torch.jit.is_scripting() and isinstance(input, list):
            # Code duplication!
            for el in input:
                print(el)
        else:
            raise Exception

module = TestModule()
scripted_module = torch.jit.script(module)

module(["1", "2"])
scripted_module(["1", "2"])

def forward(self, input: Any) -> None:
    if (torch.jit.is_scripting() and isinstance(input, List[str])) or (
        not torch.jit.is_scripting() and isinstance(input, list)
    ):  # Unfortunately this doesn't work in scripting...
        for el in input:
            print(el)