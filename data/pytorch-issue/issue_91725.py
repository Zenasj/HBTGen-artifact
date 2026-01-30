import torch

def foo(x, l):
    assert len(x.shape) == 3
    n_elem = x.shape[0]
    tmp = (
        torch.eye(4, dtype=x.dtype, device=x.device)
        .unsqueeze(0)
        .repeat(n_elem, 1, 1)
    )
    tmp = tmp.reshape(n_elem, 4, 4)
    f = x[..., 0, 0]

    tmp[..., 2, 2] = f / l

    return tmp

tmp[..., 2, 2] = f / l