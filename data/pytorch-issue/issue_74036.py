import torch.nn as nn

from torch import nn

class MyMixin:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        print("MyMixin.__init__")

class MyModuleWithMixinBefore(MyMixin, nn.Module):
    def __init__(self):
        super().__init__()
        print("MyModuleWithMixinBefore.__init__")

class MyModuleWithMixinAfter(nn.Module, MyMixin):
    def __init__(self):
        super().__init__()
        print("MyModuleWithMixinAfter.__init__")

module1 = MyModuleWithMixinBefore()
print(*(f" - {i.__module__}.{i.__name__}" for i in MyModuleWithMixinBefore.__mro__), sep="\n")

print()

module2 = MyModuleWithMixinAfter()
print(*(f" - {i.__module__}.{i.__name__}" for i in MyModuleWithMixinAfter.__mro__), sep="\n")