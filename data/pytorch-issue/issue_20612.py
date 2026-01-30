import torch

scalar = torch.tensor(5)
var1 = torch.randn(4,2,requires_grad=True)
var2 = var1.detach().requires_grad_()
output1 = var1 * scalar
output2 = var2 * scalar
output1.sum().backward()
scalar.add_(5, 1)
output2.sum().backward()
print(var1.grad)
print(var2.grad)

tensor([[5., 5.],
        [5., 5.],
        [5., 5.],
        [5., 5.]])
tensor([[10., 10.],
        [10., 10.],
        [10., 10.],
        [10., 10.]])