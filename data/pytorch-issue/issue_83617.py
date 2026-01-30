import torch

def register_multi_grad_hook(tensors, fn):
    count = 0
    nb_calls = len(tensors)
    buffer = [None] * nb_calls

    def get_inner_hook(idx):
        def inner_hook(grad):
            nonlocal count
            buffer[idx] = grad
            count += 1

            if count == nb_calls:
                fn(buffer)
        return inner_hook
    for i, t in enumerate(tensors):
        t.register_hook(get_inner_hook(i))


t1 = torch.rand(2, requires_grad=True)
t2 = torch.rand(2, requires_grad=True)
t3 = torch.rand(2, requires_grad=True)
t4 = torch.rand(2, requires_grad=True)

def hook(grads):
    print(f"Multi-hook called with {len(grads)} gradients")
register_multi_grad_hook((t2, t3), hook)

def get_hook(name):
    def hook(grad):
        print(f"{name} hook called")
    return hook
t1.register_hook(get_hook("t1"))
t3.register_hook(get_hook("t3"))


out = t1.clone()
out = out + t2
out = out + t3
out = out + t4

out.sum().backward()

import torch

def register_multi_grad_hook(tensors, fn):
    count = 0
    nb_calls = None
    buffer = None

    def get_grad_fn(t):
        # or grad accumulator
        if t.requires_grad and t.grad_fn is None:
            return t.clone().grad_fn.next_functions[0][0]
        else:
            return t.grad_fn

    grad_fns = list(map(get_grad_fn, tensors))

    def get_inner_hook(idx):
        def inner_hook(grad):
            nonlocal count, nb_calls, buffer

            if count == 0:
                # On the first call, compute the actual nb_calls and buffer
                nb_calls = sum(1 for g in grad_fns if torch._C._will_engine_execute_node(g))
                buffer = [None] * nb_calls

            buffer[idx] = grad
            count += 1

            if count == nb_calls:
                fn(buffer)
        return inner_hook
    for i, t in enumerate(tensors):
        t.register_hook(get_inner_hook(i))


t1 = torch.rand(2, requires_grad=True)
t2 = torch.rand(2, requires_grad=True)
t3 = torch.rand(2, requires_grad=True)
t4 = torch.rand(2, requires_grad=True)

def hook(grads):
    print(f"Multi-hook called with {len(grads)} gradients")
register_multi_grad_hook((t2, t3), hook)

def get_hook(name):
    def hook(grad):
        print(f"{name} hook called")
    return hook
t1.register_hook(get_hook("t1"))
t3.register_hook(get_hook("t3"))


out = t1.clone()
out = out + t2
out = out + t3
out = out + t4

out.sum().backward(inputs=(t2, t3))
# t3 hook called
# Multi-hook called with 2 gradients
# t1 hook called

out.sum().backward(inputs=(t1, t2))
# Multi-hook called with 1