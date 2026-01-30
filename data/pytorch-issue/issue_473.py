import torch
from torch.autograd import *
x = Variable(torch.zeros(5, 5), requires_grad=True)
print(x)
y = Variable(torch.range(1, 25).view(5, 5), requires_grad=True)
print(y)
idx = Variable(torch.LongTensor([0, 0, 0, 0, 0]))
z = x.index_copy(0, idx, y)
print(z)  # Note only the last row of y is copied to the first row of z. No other rows of y are used.
z.backward(torch.ones(5, 5))
print(y.grad)  # Incorrectly all ones. Only the last row of y.grad should be non-zero.