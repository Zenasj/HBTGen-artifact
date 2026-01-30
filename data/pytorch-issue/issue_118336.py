import torch
from torch.autograd.gradcheck import gradcheck

a = torch.eye(2, dtype=torch.cdouble, requires_grad=True)
gradcheck(torch.linalg.qr, (a,))