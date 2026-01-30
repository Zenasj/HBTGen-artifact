import torch

device = "cuda"
dtype = torch.float64

# Basic test for negative view
x = torch.randn(20, 20, device=device, dtype=dtype, requires_grad=False)
physical_neg = torch.neg(x)
view_neg = torch._neg_view(x)

assert torch.is_neg(view_neg), "view_neg should be negative"
assert not torch.is_neg(x), "x should not be negative"
assert torch.allclose(
    physical_neg, view_neg
), "physical_neg and view_neg should be equal"

# Test in-place operations on negative view
x = torch.randn(20, 20, device=device, dtype=dtype, requires_grad=False)
neg_x = torch._neg_view(x)
neg_x.add_(1.0)
assert torch.is_neg(neg_x), "neg_x should still be negative after in-place operation"
expected = -x + 1.0
assert torch.allclose(neg_x, expected), "neg_x should match expected result"