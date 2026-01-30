import torch

def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    empty_like_default = torch.ops.aten.empty_like.default(arg0, memory_format = torch.contiguous_format)
    bernoulli__float = torch.ops.aten.bernoulli_.float(empty_like_default);  empty_like_default = None
    div__scalar = torch.ops.aten.div_.Scalar(bernoulli__float, 0.5);  bernoulli__float = None
    mul_tensor = torch.ops.aten.mul.Tensor(arg0, div__scalar);  arg0 = div__scalar = None
    return pytree.tree_unflatten([mul_tensor], self._out_spec)

def orig_graph(x):
    return add(x, 1)

def pattern_graph(x):
    a = mul(a, x)
    return add(x, a)