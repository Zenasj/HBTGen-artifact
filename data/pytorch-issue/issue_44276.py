import torch

class TestFunc(Function):
    @staticmethod
    def forward(ctx, tensors):
        return tensors

    @staticmethod
    def backward(ctx, *grad_outputs):
        print(grad_outputs)
        tensors = [t + 1 for t in grad_outputs]
        return (tuple(tensors),)


tensors = (torch.ones(1, 5), torch.ones(1, 5), torch.ones(1, 5))

tensors[0].requires_grad = True
tensors[1].requires_grad = True
tensors[2].requires_grad = True

a, b, c = TestFunc.apply(tensors)

a = torch.sin(a) + 1
b = torch.cos(b) + 2
c = torch.tan(c) + 20

x = a.sum() + b.sum() + c.sum()
x.backward()
print(tensors[0].grad)
print(tensors[1].grad)
print(tensors[2].grad)

# Flatten arguments to give to custom Function
# only one level of tuples will be flattened
def flatten_args(*args):
    sizes = []
    res = []
    for arg in args:
        # You can handle other things than tuples
        # if needed
        if isinstance(arg, tuple):
            for el in arg:
                res.append(el)
            sizes.append(len(arg))
        else:
            res.append(arg)
            sizes.append(-1)

    return res + [sizes,]

# Unflatten arguments that was flattened above
def unflatten_args(*args):
    sizes = args[-1]
    args = args[:-1]
    res = []
    curr_idx = 0
    for size in sizes:
        if size == -1:
            res.append(args[curr_idx])
            curr_idx += 1
        else:
            res.append(tuple(args[curr_idx:curr_idx+size]))
            curr_idx += size

    return res


args = (1, 2, (3, 4), 5, 6)
print(args)
new_args = flatten_args(*args)
print(new_args)
recovered_args = unflatten_args(*new_args)
print(recovered_args)