import torch.nn as nn

import torch
import torch._dynamo as dynamo


dynamo.reset()

def toy_example():
    def add_one(x):
        return x + 1

    t = torch.nn.Parameter(torch.tensor(1.))
    t.add_one = add_one
    return t.add_one(t)

compiled_fn = torch.compile(toy_example, backend="inductor")

r1 = compiled_fn()
print(f"r1 = {r1}")

if callable(real_value):
            # Callables have more nuanced handling, and we should let the existing system delegate here.
            # Raising was past behavior and so should always be sound to fall back.
            # Note - at a certain point we may want to handle
            raise NotImplementedError