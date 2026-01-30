import torch

def evaluate_derivatives_andrew(model, s, pts):
    pts = pts.clone().detach()
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    grad_weights = torch.ones(pts.shape[0], 1)
    if is_cuda:
        pts = pts.cuda()
        model = model.cuda()
        grad_weights = grad_weights.cuda()
    elif is_mps:
        pts = pts.to('mps')
        model = model.to('mps')
        grad_weights = grad_weights.to('mps')

    pts.requires_grad_(True)
    outs = model(pts)
    grad = torch.autograd.grad(outs, pts, grad_outputs=grad_weights, create_graph=True)[0]
    return grad[:, s].detach()

factors = torch.from_numpy(variables)
if is_cuda:
    factors = factors.cuda()
elif is_mps:
    factors = factors.to('mps')
else:
    factors = factors
factors = factors.float()