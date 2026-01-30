import torch

def impl_or(ifm):
    res = getattr(ifm, '__or__')(other=78)
    return res

if __name__ == "__main__":
    x = torch.rand([10]).to(dtype=torch.uint8)
    coml_fn = torch.compile(impl_or)
    res = coml_fn(x)
    print(res)

x = torch.rand([10]).to(dtype=torch.uint8)
with torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True):
    x | 78