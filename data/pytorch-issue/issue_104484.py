import torch.nn as nn

import sys
import abc
from typing import List
import torch
import inspect

class BaseModule(torch.nn.Module, metaclass=abc.ABCMeta):
    state: List[int]

    @abc.abstractmethod
    def forward(self, x):
        """
        docstring
        """

def do_something_with_list(x: List[int]):
    if x:
        return x[-1]
    return 5

class Submodule(BaseModule):
    def __init__(self, self_x_value):
        super().__init__()
        self.x = self_x_value
        self.state = []

    def forward(self, x):
        return self.x + x + do_something_with_list(self.state)

class LowestModule(Submodule):
    def __init__(self):
        super().__init__(123)

mod = LowestModule()
mod2 = LowestModule()
torch.jit.script(mod)
torch.jit.script(mod2)

import sys
import abc
from typing import List
import torch
import inspect

class BaseModule(torch.nn.Module, metaclass=abc.ABCMeta):
    state: List[int]

    @abc.abstractmethod
    def forward(self, x):
        """
        docstring
        """

def do_something_with_list(x: List[int]):
    if x:
        return x[-1]
    return 5

class Submodule(BaseModule):
    def __init__(self, self_x_value):
        super().__init__()
        self.x = self_x_value
        self.state = []

    def forward(self, x):
        return self.x + x + do_something_with_list(self.state)

class LowestModule(Submodule):
    def __init__(self):
        super().__init__(123)

mod = LowestModule()
print(inspect.get_annotations(mod))  # prints {'state': List[int]}
inspect.getmembers(type(mod))
print(inspect.get_annotations(mod))  # prints {}