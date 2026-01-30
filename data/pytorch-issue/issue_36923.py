import torch

t = 0
x = torch.ones(()).requires_grad_()
y = t * (x / t)  # just an example; anything that produces nan's works
z = torch.where(x >= t, x, y)
z.backward()

# the forward pass works fine (the `nan`'s in `y` do not affect z)
# NOTE: this is unlike a naive implement of where that does `cond * x + (1 - cond) * y`
print(z)
# tensor(1., grad_fn=<SWhereBackward>)

# but the backward pass backprops the `nan`'s from y into x, even though the y path is never taken in torch.where
print(x.grad)
# tensor(nan)

print(x.grad)
# tensor(1.)