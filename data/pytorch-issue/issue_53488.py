import torch


def testCase3a():
    CHANNELS = 18
    EPS = 1e-5

    input = torch.rand(CHANNELS, 16 * 24 * 12)
    input[:8] -= 0.5
    input[8:] += 0.5

    x = input
    assert x.size() == (CHANNELS, 16 * 24 * 12)
    numel = x.size(1)
    assert numel == 16 * 24 * 12

    x_s = x.sum(dim=1)
    x_mean = x_s / numel
    x = (input - x_mean.unsqueeze(-1))

    W = torch.ones(CHANNELS, requires_grad=True)
    b = torch.zeros(CHANNELS, requires_grad=True)

    output = x * W.unsqueeze(-1) + b.unsqueeze(-1)
    loss = output.sum()
    loss.backward()
    my_grad = x.sum(-1)

    assert torch.allclose(W.grad, my_grad)


def testCase3b():
    CHANNELS = 18
    EPS = 1e-5

    input = torch.rand(CHANNELS, 16, 24, 12)
    input[:8] -= 0.5
    input[8:] += 0.5

    x = input
    assert x.size() == (CHANNELS, 16, 24, 12)
    x = x.reshape(CHANNELS, -1)
    assert x.size() == (CHANNELS, 16 * 24 * 12)
    numel = x.size(1)
    assert numel == 16 * 24 * 12

    x_s = x.sum(dim=1)
    x_mean = x_s / numel
    x = (input - x_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

    W = torch.ones(CHANNELS, requires_grad=True)
    b = torch.zeros(CHANNELS, requires_grad=True)

    output = x * W.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    loss = output.sum()
    loss.backward()
    my_grad = x.sum(-1).sum(-1).sum(-1)

    assert torch.allclose(W.grad, my_grad)


if __name__ == '__main__':
    testCase3a()
    testCase3b()