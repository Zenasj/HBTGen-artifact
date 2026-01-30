import torch

z = torch.rand((1, 10))

interpolation = torch.zeros(10)
for i in range(z.size(1)):
    print(i)
    interp_z = z.clone().to("mps") # does not error when .to("cpu")
    interp_z[:, i] = interpolation[i]