import torch
#import torch._dynamo   # uncommenting this solves the issue
import scipy

class MyInv_class(torch.autograd.Function):

  @staticmethod
  def forward(M):
    # M is (n,n), matrix to invert
    Mnp = M.numpy()
    Mnp_inv = scipy.linalg.inv(Mnp)
    M_inv = torch.from_numpy(Mnp_inv)
    return M_inv
  
  @staticmethod
  def setup_context(ctx, inputs, outputs):
    M = inputs
    M_inv = outputs
    ctx.save_for_backward(M_inv)
    ctx.M_inv = M_inv

  @staticmethod
  def backward(ctx, A):
    Minv = ctx.saved_tensors[0]
    return - Minv.T @ A @ Minv.T
  
  @staticmethod
  def jvp(ctx, N):
    Minv = ctx.M_inv
    return - Minv @ N @ Minv

MyInv = MyInv_class.apply

# testing:

n = 10
M = torch.rand((n,n), requires_grad=True)

N = torch.rand((n,n))
df = torch.func.jvp(MyInv,(M,),(N,))[0]    # error appears at this line
df_torch = torch.func.jvp(torch.linalg.inv,(M,),(N,))[0]
print((torch.norm(df-df_torch)/torch.norm(df_torch)).item())