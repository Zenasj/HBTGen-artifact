import torch

@torch.library.custom_op("mylib::mysin", mutates_args=["out_list"], schema="(Tensor x, Tensor(a!)[]? out_list) -> (Tensor)")
def mysin(x: torch.Tensor, out_list: list[torch.Tensor] = None) -> torch.Tensor:
    r = x.sin()
    return r

@torch.library.register_fake("mylib::mysin")
def mysin_fake(x, out_list: list[torch.Tensor] = None) -> torch.Tensor:
    return torch.empty_like(x)


def fn(x):
    x = x * 3
    s = [torch.empty_like(x)]
    x= mysin(x, out_list=s)
    x =  x / 3
    return x

fn = torch.compile(fn)

x = torch.randn(3, requires_grad=False)
y= fn(x)
print(y)