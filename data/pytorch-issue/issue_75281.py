import torch

x = torch.empty(4, 5)
y = torch.empty(5, 4)

generator = torch.Generator()
orig_state = generator.get_state()

x.normal_(generator=generator)

y = y.transpose(0, 1)
generator = generator.set_state(orig_state)
y.normal_(generator=generator)
assert torch.allclose(x, y)  # fails