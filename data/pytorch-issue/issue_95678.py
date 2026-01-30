import torch

class DDE_adjoint(torch.autograd.Function):
    def forward(ctx, X, x0, tau, T, *parameters):
        ctx.X = X
        ctx.x0 = x0
        ctx.tau = tau
        ctx.T = T
        with torch.no_grad():
            ans = dde_solve(X, x0, tau, T)
            ctx.save_for_backward(ans, *parameters)
        return ans.clone()
        
 
    def backward(ctx, grad_y):
        print('custom backward called')
        return None 

solution = DDE_adjoint.apply
sol = solution(v_field, torch.tensor(1, dtype = float, requires_grad = True), torch.tensor(1,dtype = float, requires_grad = True), torch.tensor(10) )
sol.backward()