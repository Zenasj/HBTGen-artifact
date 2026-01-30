import torch                                                                                                                                      
from torch.autograd import gradcheck, gradgradcheck                                                                                               
                                                                                                                                                  
class LU(torch.autograd.Function):                                                                                                                
    @staticmethod                                                                                                                                 
    def forward(ctx, A):                                                                                                                          
        lu, pivots = A.lu()                                                                                                                       
        p, l, u = torch.lu_unpack(lu, pivots)                                                                                                     
        ctx.save_for_backward(l, u, p)                                                                                                            
        ctx.mark_non_differentiable(pivots)
        return lu, pivots

    @staticmethod
    def backward(ctx, LU_grad, pivots_grad):
        L, U, P = ctx.saved_tensors

        Ltinv = L.inverse().transpose(-1, -2)
        Utinv = U.inverse().transpose(-1, -2)

        phi_l = (L.transpose(-1, -2) @ LU_grad).tril_()
        phi_l.diagonal(dim1=-2, dim2=-1).mul_(0.0)

        phi_u = (LU_grad @ U.transpose(-1, -2)).triu_()

        A_grad = Ltinv @ (phi_l + phi_u) @ Utinv

        return P @ A_grad, None

for i in range(5):
    print(i)
    size = 2
    S = torch.rand(2, size, size, dtype=torch.double).requires_grad_(True)

    gradcheck(LU.apply, S)
    gradgradcheck(LU.apply, S, [torch.rand(2, size, size, dtype=torch.double)])