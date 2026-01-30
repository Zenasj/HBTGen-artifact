import torch.nn as nn

import torch

def case1():
    linear = torch.nn.Linear(3, 5)
    optimizer = torch.optim.Adagrad(linear.parameters())

    x = torch.randn(2, 3)
    y = linear(x)

    loss = y.sum()
    loss.backward()

    optimizer.step()


def case2():
    linear = torch.nn.Linear(3, 5)
    optimizer = torch.optim.Adagrad(linear.parameters())

    linear2 = torch.nn.Linear(5, 4)
    optimizer.add_param_group({'params': [linear2.weight, linear2.bias]})

    x = torch.randn(2, 3)
    y1 = linear(x)
    y2 = linear2(y1)

    loss = y2.sum()
    loss.backward()

    optimizer.step()


if __name__ == '__main__':
    case1()
    case2()