import torch

class Functional1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, y0, *params):
        ctx.fcn = fcn
        ctx.save_for_backward(y0, *params)
        return y0
    @staticmethod
    def backward(ctx, grad_yout):
        yout, *params = ctx.saved_tensors
        with torch.enable_grad():
            params_copy = [p.clone().requires_grad_() for p in params]
            yfcn = ctx.fcn(yout, *params_copy)
        # NOTE: setting create_graph to False removes the memory leak
        grad_params = torch.autograd.grad(yfcn, params_copy, grad_outputs=grad_yout,
                                          create_graph=torch.is_grad_enabled())
        return (None, None, *grad_params)

def fcn2(x, x2):
    return x + x2

def test_fcn():
    shape = (10000, 10000)
    y0 = torch.zeros(shape).to(torch.double).to(torch.device("cuda"))
    y2 = y0.clone().requires_grad_()
    loss = (Functional1.apply(fcn2, y0, y2) ** 2).sum()
    # NOTE: setting create_graph to False removes the memory leak, even if
    # retain_graph is still True
    loss.backward(retain_graph=True, create_graph=True)

for i in range(5):
    test_fcn()
    torch.cuda.empty_cache()
    mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
    print("memory allocated:", mem, "MiB")