import torch

torch.cuda.device(device)

opts.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu") if opts.debug else opts.gpu

requirements = (CUDADeviceName != "GeForce GTX TITAN X")