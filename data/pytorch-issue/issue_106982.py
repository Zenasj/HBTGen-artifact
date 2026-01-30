import torch

class tensorExtra(torch.Tensor):
    ...

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        output = super().__torch_function__(func, types, args, kwargs)
        if func is torch.stack:
            # store a list of extra attributes from each of the stack args
            output.extra = [arg.extra for arg in args[0] if isinstance(arg, tensorExtra)]
        return output

stacked = torch.stack([tensorExtra('abc'), tensorExtra('def')])
print(stacked.extra)  # ['abc', 'def']