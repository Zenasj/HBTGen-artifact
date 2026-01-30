import torch


def test_foo():
    @torch.compile
    def foo():
        def inner(a, b, res_dtype):
            print(a, b, res_dtype)
            assert torch.result_type(a, b) == res_dtype
        inner(torch.tensor(1, device="cpu"), 1., torch.get_default_dtype()) 
    torch.set_default_dtype(torch.float)
    foo()
    torch.set_default_dtype(torch.double)
    foo()

test_foo()