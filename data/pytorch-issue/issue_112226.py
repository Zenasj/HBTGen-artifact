import random

import torch
def model(shape, generator):
      return torch.randn([1, 4, 64, 64], generator=generator, device="cuda:0")
model = torch.compile(model)
x = model((1, 3, 64, 64), None)
print(x)

@register_lowering(aten.randn)
def randn(*args, **kwargs):
    if kwargs.get("generator", None) is not None:
        return fallback_randn_generator(*args, **kwargs)
    elif config.fallback_random:
        return fallback_randn_default(*args, **kwargs)
    raise AssertionError("should have been handled in replace_random.py")

# fallback to eager for random/dropout, this is slow but useful for debugging
fallback_random = True