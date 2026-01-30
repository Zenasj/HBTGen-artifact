import torch
from functools import partial

#DEVICE = "cpu"
DEVICE = "cuda"

def square():
    A = torch.rand(3, 3, device=DEVICE)
    A = A.T @ A  + torch.eye(3, device=DEVICE)
    return A

def warn(fun, n_args, has_out=True):
    args = [square() for _ in range(n_args)]
    print(40*"-" + "  NORMAL {}  ".format(fun.__name__ if hasattr(fun, "__name__") else "") + 40*"-")
    out = fun(*args)
    if has_out:
        print(40*"-" + "  OUT {}  ".format(fun.__name__ if hasattr(fun, "__name__") else "") + 40*"-")
        fun(*args, out=out)
    input()
    print()
    print()

# Shows one by one the warnings (press enter)
# The script should be run once with each of the possibilities for the global variable DEVICE 
warn(torch.cholesky, 1)
warn(torch.eig, 1, has_out=False)     # Warn in common
warn(torch.symeig, 1, has_out=False)  # Warn in common
warn(torch.svd, 1)
warn(torch.qr, 1)
warn(torch.matrix_rank, 1, has_out=False)
warn(partial(torch.matrix_rank, tol=1e-4), 1, has_out=False)  # Has a different path
warn(torch.chain_matmul, 2)
warn(torch.solve, 2)
warn(torch.lstsq, 2)

import torch
from functools import partial
from itertools import product

def square(device):
    for size in ((2, 3, 3), (3, 3)):
        A = torch.rand(*size, device=device)
        yield A.transpose(-2, -1) @ A  + torch.eye(3, device=DEVICE)

def non_square(device):
    for size in ((2, 3, 4), (2, 4, 3), (3, 4), (4, 3), (2, 3, 3), (3, 3)):
        yield torch.rand(*size, device=device)

def assert_eq(f1, f2, *args):
    X1 = f1(*args)
    X2 = f2(*args)
    if isinstance(X1, tuple):
        for t1, t2 in zip(X1, X2):
            if not torch.allclose(t1, t2, atol=1e-3, rtol=1e-4):
                print(t1, t2)
                raise RuntimeError(f"{f1.__name__}\n{f2.__name__}\n{args}")
    else:
        if not torch.allclose(X1, X2, atol=1e-3, rtol=1e-4):
            print(X1, X2)
            raise RuntimeError(f"{f1.__name__}\n{f2.__name__}\n{args}")


def cholesky(device):
    def my_cholesky(A, upper):
        if upper:
            return torch.linalg.cholesky(A.transpose(-2, -1)).transpose(-2, -1)
        else:
            return torch.linalg.cholesky(A)

    for t, upper in product(square(device), [True, False]):
        assert_eq(torch.cholesky, my_cholesky, t, upper)


def symeig(device):
    def my_symeig(A, eigenvectors):
        if eigenvectors:
            L, V = torch.linalg.eigh(A)
            return L, V.abs()
        else:
            return torch.linalg.eigvalsh(A)

    def torch_symeig(A, eigenvectors):
        if eigenvectors:
            L, V = torch.symeig(A, eigenvectors)
            return L, V.abs()
        else:
            return torch.symeig(A, eigenvectors)[0]

    for t, eigenvectors in product(square(device), [True, False]):
        assert_eq(torch_symeig, my_symeig, t, eigenvectors)

def svd(device):
    def my_svd(A, some, compute_uv):
        if compute_uv:
            U, S, Vh = torch.linalg.svd(A, full_matrices=not some)
            return U.abs(), S, Vh.transpose(-2, -1).conj().abs()
        else:
            S = torch.linalg.svdvals(A)
            return S

    def torch_svd(A, some, compute_uv):
        if compute_uv:
            U, S, V = torch.svd(A, some, True)
            return U.abs(), S, V.abs()
        else:
            _, S, _ = torch.svd(A, some, True)
            return S

    for t, some, compute_uv in product(non_square(device), [True, False], [True, False]):
        assert_eq(torch_svd, my_svd, t, some, compute_uv)

def qr(device):
    def my_qr(A, some):
        return torch.linalg.qr(A, "reduced" if some else "complete")

    for t, some in product(non_square(device), [True, False]):
        assert_eq(torch.qr, my_qr, t, some)


def matrix_rank(device):
    def my_matrix_rank(A, symmetric):
        return torch.linalg.matrix_rank(A, hermitian=symmetric)

    for t, symmetric in product(square(device), [True, False]):
        assert_eq(torch.matrix_rank, my_matrix_rank, t, symmetric)

def chain_matmul(device):
    def my_multi_dot(*tensors):
        return torch.linalg.multi_dot(tensors)

    make_t = partial(torch.rand, device=device)
    assert_eq(torch.chain_matmul, my_multi_dot, make_t(3,2), make_t(2,3))
    assert_eq(torch.chain_matmul, my_multi_dot, make_t(3,3), make_t(3,3))

def solve(device):
    def my_solve(B, A):
        return torch.linalg.solve(A, B)

    def torch_solve(B, A):
        return torch.solve(B, A).solution

    for t in square(device):
        assert_eq(torch_solve, my_solve, t, t.clone())


def lstsq(device):
    def my_lstsq(B, A):
        return torch.linalg.lstsq(A, B).solution

    def torch_lstsq(B, A):
        return torch.lstsq(B, A).solution[:A.size(1)]

    for t in non_square(device):
        if t.ndim == 3: # torch.lstsq cuda does not support batches
            continue
        if DEVICE == "cuda" and t.size(-2) < t.size(-1): # Not implemented
            continue
        assert_eq(torch_lstsq, my_lstsq, t, t.clone())


# If none of them throws, we're good

# eig is too different, we will not compare it
for device in ["cpu", "cuda"]:
    cholesky(device)
    symeig(device)
    svd(device)
    qr(device)
    matrix_rank(device)
    chain_matmul(device)
    solve(device)
    lstsq(device)