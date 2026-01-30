import torch


with torch.autograd.set_detect_anomaly(True):
    a = torch.rand(1, requires_grad=True)
    c = torch.rand(1, requires_grad=True)

    b = a ** 2 * c ** 2
    b += 1
    b *= c + a

    d = b.exp_()
    d *= 5

    b.backward()