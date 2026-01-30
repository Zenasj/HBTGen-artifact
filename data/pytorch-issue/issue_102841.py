import torch

def test_setitem_boolean_mask_diff(self):
        def fn(x, b, y):
            x = x.clone()
            x[b] = y
            return x

        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        x = torch.randn(4, requires_grad=True)
        b = torch.tensor([True, False, True, False])
        y = torch.randn(2, requires_grad=True)
        opt_fn(x, b, y)