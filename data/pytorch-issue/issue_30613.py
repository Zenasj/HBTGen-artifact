import torch

torch.div(torch.ones(1), torch.zeros(1))
# tensor([inf])
torch.div(torch.tensor([1]), torch.tensor([0]))
# Process finished with exit code 136 (interrupted by signal 8: SIGFPE)