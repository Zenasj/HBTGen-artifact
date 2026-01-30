import torch
convin = torch.load('convin.pt')
coefficients = torch.load('input.pt')
convin(coefficients)

tensor([   nan,    nan,    nan, 0.0518], requires_grad=True)