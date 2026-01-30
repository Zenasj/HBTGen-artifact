import torch

x = torch.ones([2, 2], requires_grad=True)
y = torch.ones([2, 2], requires_grad=True)
z = torch.ones([2, 2], requires_grad=True)

def helper_fn(x, y, z):
    return x * y * z

out = torch._dynamo.export(helper_fn)(x, y, z)
print(out.graph_module)

@torch._dynamo.assume_constant_result
def helper_fn2(x, y, z):
    return x * y * z

out = torch._dynamo.export(helper_fn2)(x, y, z)
print(out.graph_module)

def helper_fn3(x, y, z):
    return helper_fn2(x, y, z)

out = torch._dynamo.export(helper_fn3)(x, y, z)
print(out.graph_module)

def helper_fn4(x, y, z):
    return helper_fn2(x, y, z) * x

out = torch._dynamo.export(helper_fn4)(x, y, z)

def forward(self, x, y, z):
    arg0, arg1, arg2, = fx_pytree.tree_flatten_spec(([x, y, z], {}), self._in_spec)
    l_x_ = arg0
    helper_fn2 = self.helper_fn2
    mul = helper_fn2 * l_x_;  helper_fn2 = l_x_ = None
    return pytree.tree_unflatten([mul], self._out_spec)