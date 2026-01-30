import torch
torch.backends.cuda.preferred_linalg_library(backend="cusolver")  # both magma and cusolver backends fail as expected
input = torch.tensor([[1.0]*2, [3]*2], requires_grad=True).cuda()
output = torch.linalg.inv(torch.matmul(input, input))
print(output)