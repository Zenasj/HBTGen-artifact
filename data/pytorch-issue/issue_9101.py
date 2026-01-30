import torch

x = numpy.zeros([10, 10])
x.shape          # (10, 10)
x[-3:0].shape    # (0, 10)
x[-3:0, :].shape # (0, 10)
x[:, -3:0].shape # (10, 0)

x = torch.zeros(10, 10)
x.shape          # torch.Size([10, 10])
x[-3:0].shape    # torch.Size([0])
x[-3:0, :].shape # error
x[:, -3:0].shape # torch.Size([10])