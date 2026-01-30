import torch as th
from scipy import special, optimize
import numpy as np
import itertools as it
th.set_default_dtype(th.float64)


def finite_difference_grad(func, x, *args, epsilon=1e-6, return_sign, **kwargs):
    """
    Compute the gradient of the first element returned by `func`, discarding signs
    if necessary.
    """
    x1 = x.detach().numpy().copy()
    x2 = x1.copy()
    x2.ravel()[0] += epsilon
    x1.ravel()[0] -= epsilon
    y2 = func(x2, *args, **kwargs, return_sign=return_sign)
    y1 = func(x1, *args, **kwargs, return_sign=return_sign)
    if return_sign:
        y2, _ = y2
        y1, _ = y1
    grad = (y2 - y1) / (2 * epsilon)
    return grad


def maybe_detach(x):
    return None if x is None else x.detach()


for return_sign, scaled, use_out in it.product(*([(False, True)] * 3)):
    print(f'return_sign={return_sign}, scaled={scaled}, use_out={use_out}')
    
    # Test for random values.
    x = (th.rand((4, 5)) - .5)
    if scaled:
        b = (th.rand(5) - .5)
    else:
        b = None
        
    if use_out:
        result_out = th.empty(4)
        sign_out = th.empty(4)
        out = (result_out, sign_out)
    else:
        out = None
        x.requires_grad_()
        if b is not None:
            b.requires_grad_()
    
    # Verify the output.
    expected = special.logsumexp(x.detach(), axis=-1, b=maybe_detach(b), return_sign=return_sign)
    actual = x.logsumexp(dim=-1, b=b, return_sign=return_sign, out=out)

    if return_sign:
        expected, expected_sign = expected
        actual, actual_sign = actual
        np.testing.assert_array_equal(actual_sign, expected_sign)
    np.testing.assert_allclose(actual.detach(), expected)

    # Skip the gradients when using out.
    if use_out:
        continue

    # Verify the gradients...
    actual[0].backward()

    # ... for the main input and ...
    actual_grad = x.grad
    expected_grad = finite_difference_grad(special.logsumexp, x, b=maybe_detach(b), 
                                           return_sign=return_sign, axis=-1)
    np.testing.assert_allclose(actual_grad.ravel()[0], expected_grad[0], rtol=1e-6)

    # ... for the scalars.
    if scaled:
        actual_grad = b.grad
        expected_grad = finite_difference_grad(
            lambda b, *args, **kwargs: special.logsumexp(x.detach(), axis=-1, b=b, return_sign=return_sign), 
            maybe_detach(b), return_sign=return_sign,
        )
        np.testing.assert_allclose(actual_grad.ravel()[0], expected_grad[0], rtol=1e-6)

import torch as th

x = th.randint(-10, 10, [5, 4])
out = th.randn(5)
th.special.logsumexp(x, 1, out=out)

def logsumexp(x, b, dim, eps=1e-12):
    b = torch.clamp(b, min=eps)
    return torch.logsumexp(x + torch.log(b), dim=dim)