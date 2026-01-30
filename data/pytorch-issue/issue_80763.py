import torch

input = torch.tensor([-0.8137-0.1476j, -0.2749-0.2630j, -0.2163-0.7010j, -0.3824-0.2827j],
       dtype=torch.complex128, requires_grad=True)

torch.autograd.gradcheck(torch.acosh, (input))