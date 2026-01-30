import torch
import torch.nn as nn

class MySparseMatMul(torch.autograd.Function):
  @staticmethod
  def forward(ctx, a, b):
    # Is the detach needed / helpful?
    ad = a.detach()
    bd = b.detach()
    x = torch.sparse.mm(ad, bd)
    if a.requires_grad or b.requires_grad:
      # Not sure if the following is needed / helpful
      x.requires_grad = True
    # Save context for backward pass
    ctx.save_for_backward(ad, bd)
    return x

  @staticmethod
  def backward(ctx, prev_grad):
    # Recover context
    a, b = ctx.saved_tensors
    # The gradient with respect to the matrix a seen as a dense matrix would
    # lead to a backprop rule as follows
    # grad_a = prev_grad @ b.T
    # but we are only interested in the gradient with respect to
    # the (non-zero) values of a. To save memory, instead of computing the full
    # dense matrix prev_grad @ b and then subsampling at the nnz locations in a,
    # we can directly only compute the required values:
    # grad_a[i,j] = dotprod(prev_grad[i,:], b[j,:])
    # We start by getting the i and j indices
    if(a.layout == torch.sparse_coo):
      grad_a_idx = a.indices()
      grad_a_row_idx = grad_a_idx[0,:]
      grad_a_col_idx = grad_a_idx[1,:]
    elif(a.layout == torch.sparse_csr):
      grad_a_col_idx = a.col_indices()
      grad_a_crow_idx = a.crow_indices()
      # uncompress row indices
      grad_a_row_idx = torch.repeat_interleave(
          torch.arange(a.size()[0], device=a.device),
          grad_a_crow_idx[1:]-grad_a_crow_idx[:-1] )
    else:
      raise ValueError(f"Unsupported layout: {a.layout}")
    
    # Get prev_grad[a_row_idx,:]
    prev_grad_select = prev_grad.index_select(0,grad_a_row_idx)
    # Get b[a_col_idx,:]
    b_select = b.index_select(0,grad_a_col_idx)
    # Element-wise multiplication
    prev_grad_b_ewise = prev_grad_select * b_select
    if b.dim() == 1:
      # if b is a vector, the dot prod and elementwise multiplication are the same
      grad_a_vals = prev_grad_b_ewise
    else:
      # if b is a matrix, the dot prod requires summation
      grad_a_vals = torch.sum( prev_grad_b_ewise, dim=1 )
    # Create a sparse matrix of the gradient with respect to the nnz of a
    if(a.layout == torch.sparse_coo):
      grad_a = torch.sparse_coo_tensor(grad_a_idx, grad_a_vals, a.shape)
    elif(a.layout == torch.sparse_csr):
      grad_a = torch.sparse_csr_tensor(grad_a_crow_idx, grad_a_col_idx,
                                       grad_a_vals, a.shape)

    # Now compute the (dense) gradient with respect to b
    grad_b = torch.t(a) @ prev_grad
    return grad_a, grad_b

my_sparse_mm = MySparseMatMul.apply

def flin_as_sparse_mm(A, B):
  return torch.nn.functional.linear(input=B, weight=A, bias=None)