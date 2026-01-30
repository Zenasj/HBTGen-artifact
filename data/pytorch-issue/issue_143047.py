py
import torch

requires_grad_ls = [False, True]
dimensions = ["0d", "1d"]
for requires_grad in requires_grad_ls:
    for dimension in dimensions:
        try:
            params = torch.tensor([1.])
            grad = torch.tensor([2.])
            lr = torch.tensor(0.1, requires_grad=requires_grad)
            if dimension == "1d":
                lr = lr.reshape(1,)
            params.add_(grad, alpha=-lr)
            print(f"success for requires_grad: {requires_grad}, dimension: {dimension}")
        except Exception as e:
            print(f"faliure for requires_grad: {requires_grad}, dimension: {dimension}, {e}")