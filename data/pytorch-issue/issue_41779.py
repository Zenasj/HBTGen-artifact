import torch

tensor([[ 9.,  2.,  9.],
        [-2., -3., -4.],
        [ 7.,  8., -9.]])

tensor([[ 1.,  0.,  1.],
        [-0., -0., -0.],
        [ 0.,  0., -1.]])

tensor([[ 0.3333,  0.0000,  0.3333],
        [-0.0000, -0.0000, -0.0000],
        [ 0.0000,  0.0000, -0.3333]])

with torch.enable_grad():
    a = torch.tensor([
        [9., 2., 9.],
        [-2., -3., -4.],
        [7., 8., -9.],
    ], requires_grad=True)
    b = torch.norm(a, p=float('inf'))
    b.backward()
    print(a.grad)

tensor([[ 0.3333,  0.0000,  0.3333],
        [-0.0000, -0.0000, -0.0000],
        [ 0.0000,  0.0000, -0.3333]])

with torch.enable_grad():
    a = torch.tensor([
        [9., 2., 9.],
        [-2., -3., -4.],
        [7., 8., -9.],
    ], requires_grad=True)
    b = torch.norm(a, p=30)
    b.backward()
    print(a.grad)

tensor([[ 3.4248e-01,  3.9037e-20,  3.4248e-01],
        [-3.9037e-20, -4.9903e-15, -2.0958e-11],
        [ 2.3413e-04,  1.1252e-02, -3.4248e-01]])

tensor([[ 0.3333,  0.0000,  0.3333],
        [-0.0000, -0.0000, -0.0000],
        [ 0.0000,  0.0000, -0.3333]])

tensor([[ 1.,  0.,  1.],
        [-0., -0., -0.],
        [ 0.,  0., -1.]])

with torch.enable_grad():
    a = torch.tensor([
        [3., 0., 4.],
        [-1., -2., -4.],
        [-5., 0., 0.],
    ], requires_grad=True)
    b = torch.norm(a, p=2, dim=1)
    print(b)
    b = torch.norm(b, p=float('inf'))
    b.backward()
    print(a.grad)