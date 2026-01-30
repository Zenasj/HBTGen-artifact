import torch

@torch.compile(fullgraph=True)
def func(A):
    # def f(a) -> Union[torch.Tensor, int]:  # => this will fail
    def f(a):                                # => this will work
        if a.ndim == 2:
            return a
        return 3
    return f(A)