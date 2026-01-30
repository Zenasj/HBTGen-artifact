import torch

x1 = torch.tensor([0.0, 2.0, 3.0], requires_grad=True)
x2 = torch.tensor([0.0, 2.0, 3.0], requires_grad=True)

# Use torch.where to conditionally apply an operation
y1 = torch.where(x1 > 2, torch.log2(x1), 2*x1)

# Equaivalent code
y2 = x2.clone().detach()
for i in range(len(x2)):
    if x2[i] > 2:
        y2[i] = torch.log2(x2[i])
    else:
        y2[i] = 2*x2[i]

z1 = y1.sum()
z2 = y2.sum()

z1.backward()
z2.backward()

# Check the gradients
print('y1 =',y1)
print('x1.grad =', x1.grad)
print('y2 =',y2)
print('x2.grad = ', x2.grad)