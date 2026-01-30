import torch

a = {'foo': torch.zeros(())}
mod = Foo()
print(a)  # {'foo': tensor(0.)}
functional_call(mod, a, torch.ones(()))
print(a)  # Should this be {'foo': tensor(0.)} or {'foo': tensor(1.)} ?