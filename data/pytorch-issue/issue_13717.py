import torch
x = torch.rand(2, 5).to(torch.device('cuda:0'))
x2 = torch.ones(3, 5).to(torch.device('cuda:0'))
x3 = torch.tensor([[1], [2]]).to(torch.device('cuda:0'))
x2.scatter_add_(1, x3, x)

x = torch.rand(2, 5).to(torch.device('cpu'))
x2 = torch.ones(3, 5).to(torch.device('cpu'))
x3 = torch.tensor([[1], [2]]).to(torch.device('cpu'))
x2.scatter_add_(1, x3, x)

tensor([[1.0000e+00, 1.2632e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00],
        [1.0000e+00, 1.0000e+00, 1.9192e+00, 1.0000e+00, 1.0000e+00],
        [1.1930e+30, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00]])