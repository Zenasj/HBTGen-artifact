import torch

x_cpu = torch.tensor([3.2827e-05+0.0000e+00j,  2.2128e-05-5.1657e-03j,
         1.3144e-04-1.0340e-02j,  3.2078e-04-1.5584e-02j,
         5.1421e-04-2.0760e-02j,  7.7687e-04-2.5986e-02j], device="cpu")


x_mps = torch.tensor([3.2827e-05+0.0000e+00j,  2.2128e-05-5.1657e-03j,
         1.3144e-04-1.0340e-02j,  3.2078e-04-1.5584e-02j,
         5.1421e-04-2.0760e-02j,  7.7687e-04-2.5986e-02j], device="mps")

print(x_cpu.abs())
print(x_mps.abs())