import torch

x = torch.ones([1], requires_grad=True) # weight
w = torch.tensor([0.2], requires_grad=True) # lr

def f(x):
    x = x.cuda()
    return torch.pow(x, 2).sum()

def SGD(grad, lr=0.2):
    return -lr*grad

def optimizer(grad):
    return -w*grad

sum_losses = 0
for i in range(3):
    loss = f(x)
    sum_losses += loss
    loss.backward(torch.ones_like(loss), retain_graph=True)
    print('x.grad: {}'.format(x.grad))
    print('w1.grad: {}'.format(w.grad))

    update = optimizer(x.grad)
    x = x + update
    print('x-:{}'.format(x))
    print('x-.grad: {}'.format(x.grad))

    x.retain_grad()
    update.retain_grad()

sum_losses.backward()
print('w.grad: {}'.format(w.grad))

w_update = SGD(w.grad, lr=0.1)
w = w + w_update
print('w====: {}'.format(w))