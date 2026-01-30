import torch


@torch.compile(backend="eager")
def fn(x, y):
    return x * y


for i in range(1, 10):
    x = torch.randn(4, i)

    # create a view for i > 1
    if i == 1:
        x1 = x
    else:
        x1 = x[:, 0:1]

    y = torch.randn(4, 1)
    print(x1.shape, y.shape)
    fn(x1, y)