import torch.nn as nn


class TestClass(nn.Module):
    def __init__(self):
        super().__init__()
        self._some_property = None
        self.indicator = "not set"

    @property
    def some_property(self):
        return self._some_property

    @some_property.setter
    def some_property(self, value):
        self.indicator = "set"
        self._some_property = value


if __name__ == "__main__":
    foo = TestClass()
    foo.some_property = "not a module"
    print(f"Result foo: {foo.indicator}")

    bar = TestClass()
    bar.some_property = nn.Module()
    print(f"Result bar: {bar.indicator}")

def __setattr__(self, name, value):
        if name in vars(type(self)) and isinstance(vars(type(self))[name], property):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

def __setattr__(self, name, value):
        # Explicitly get vars from all base classes (via the method resolution attribute __mro__)
        all_vars = {}
        for base_class in type(self).__mro__:
            all_vars.update(vars(base_class))

        if name in all_vars and isinstance(all_vars[name], property):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

from torch.nn import Module

class Foo(Module):
    def __init__(self, bar = None):
        super().__init__()
        self._bar = bar

    @property
    def bar(self):
        return self._bar
    
    @bar.setter
    def bar(self, bar):
        self._bar = bar


foo = Foo()
foo.bar = Module()