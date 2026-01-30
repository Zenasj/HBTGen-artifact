import torch
test = torch.tensor([0.+0.3071j], dtype=torch.complex64, requires_grad=True)
torch.autograd.gradcheck(torch.angle, (test))

# torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
# numerical:tensor([[-3.2783+0.j]])
# analytical:tensor([[-3.2563+0.j]])