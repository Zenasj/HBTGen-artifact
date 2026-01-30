import torch

# two scalars
torch.ones(()) + torch.ones(()).cuda()  # OK
torch.ones(()).cuda() + torch.ones(())  # OK

# one scalar (CPU), one vector (GPU)
torch.ones(()) + torch.ones(1).cuda()  # OK
torch.ones(1).cuda() + torch.ones(())  # OK

# one scalar (GPU), one vector (CPU)
torch.ones(()).cuda() + torch.ones(1)  # fails
torch.ones(1) + torch.ones(()).cuda()  # fails