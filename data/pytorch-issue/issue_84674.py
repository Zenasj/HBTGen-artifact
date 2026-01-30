import torch

def f(x):
    return x.t()

x = torch.randn(2, requires_grad=True)
y = f(x)

compiled_f = make_fx(f)(x)
y_compiled = compiled_f(x)


print(compiled_f)
print("y.requires_grad", y.requires_grad)
print("y_compiled.requires_grad", y_compiled.requires_grad)


# def forward(self, x_1):
#     t_default = torch.ops.aten.t.default(x_1);  x_1 = None
#     detach_default = torch.ops.aten.detach.default(t_default);  t_default = None
#     return detach_default

# y.requires_grad True
# y_compiled.requires_grad False