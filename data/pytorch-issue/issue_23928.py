import random

import time
import torch
import numpy as np

def _batch_trtrs_lower(bb, bA):
    """
    Applies `torch.trtrs` for batches of matrices. `bb` and `bA` should have
    the same batch shape.
    """
    flat_b = bb.reshape((-1,) + bb.shape[-2:])
    flat_A = bA.reshape((-1,) + bA.shape[-2:])
    flat_X = torch.stack([torch.trtrs(b, A, upper=False)[0] for b, A in zip(flat_b, flat_A)])
    return flat_X.reshape(bb.shape)


def _batch_mahalanobis_old(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.
    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bL = bL.expand(bx.shape[bx.dim() - bL.dim() + 1:] + (n,))
    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = _batch_trtrs_lower(flat_x_swap, flat_L).pow(2).sum(-2)  # shape = b x c
    return M_swap.t().reshape(bx.shape[:-1])


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.
    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.
    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.
    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.triangular_solve(flat_x_swap, flat_L, upper=False)[0].pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


input_1 = torch.from_numpy(np.linalg.cholesky(np.diag(np.random.rand(2).astype(np.float32)))).cuda()
input_2 = torch.from_numpy(np.random.rand(13440, 2).astype(np.float32)).cuda()

runs = 1000

total_time_new, total_time_old = 0, 0
relative_error_cum = 0
for _ in range(runs):
    torch.cuda.synchronize()
    st = time.perf_counter()
    m_new = _batch_mahalanobis(input_1, input_2)
    torch.cuda.synchronize()
    total_time_new += time.perf_counter() - st

    st = time.perf_counter()
    m_old = _batch_mahalanobis_old(input_1, input_2)
    torch.cuda.synchronize()
    total_time_old += time.perf_counter() - st

    relative_error_cum = torch.norm(m_new-m_old) / torch.norm(m_new)
    
print(f'[New] Time Taken: {total_time_new:.6f}s')
print(f'[Old] Time Taken: {total_time_old:.6f}s')
print(f'Average Relative Error: {relative_error_cum/runs}')