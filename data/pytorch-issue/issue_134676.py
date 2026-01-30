import torch

def raw_function(t0, t1, t2):
    t = t0.shape
    t0 = torch.relu(t0)
    shape = (t[0], int(t[1] * t[2]))
    reshape_t = t0.reshape(shape)
    o1 = torch.add(reshape_t, t1)
    o2 = torch.add(t2, t2)
    grad = torch.ones_like(o2)
    o2.backward(grad)
    return o1, o2

torch._dynamo.config.compiled_autograd = True
compiled_fn = torch.compile(raw_function, backend="eager", dynamic=True)

for s in input_shapes:
    t0 = torch.randn(s[0], requires_grad=True)
    t1 = torch.randn(s[1], requires_grad=True)
    t2 = torch.randn(s[2], requires_grad=True)
    o1, o2 = compiled_fn(t0, t1, t2)