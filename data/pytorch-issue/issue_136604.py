import torch

torch.cuda.set_device(4)  # on the other side, 5

device_mesh = DeviceMesh("cuda", mesh=[0, 1])

tensor = torch.ones(1).cuda()

tensor([1.], device='cuda:0')

device_mesh = DeviceMesh("cuda", mesh=[0, 1])
torch.cuda.set_device(4)
tensor = torch.ones(1).cuda()