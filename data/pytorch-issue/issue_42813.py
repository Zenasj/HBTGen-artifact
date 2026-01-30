import torch

In [45]: t1 = torch.rand(2, 100)

In [46]: t2 = torch.stft(t1, 10)

In [47]: t3 = torch.istft(t2, 10)

In [48]: t3.shape
Out[48]: torch.Size([2, 100])

X = X.reshape(-1, shape[-3], shape[-2], shape[-1])

y = torch.istft(
    X,
    # ...
)

y = y.reshape(shape[:-3] + y.shape[-1:])