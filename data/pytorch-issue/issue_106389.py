import torch

def test_grad_pytree(self):
    ...
    def fn(x):
        x1, x2 = x
        return x1.sin().sum() + x2

    def wrapper_fn(x):
        return torch.func.grad(fn)(x)

    expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)((x1, x2))