import torch.nn as nn
import random

import numpy as np
import torch
import torch.nn.functional as F


def compute_grad(a, gt, device):
    a_t = torch.tensor(a, requires_grad=True, device=device)
    # a_t.retain_grad()

    gt_t = torch.tensor(gt).to(device)

    criterion = torch.nn.BCELoss()

    # shape is equal
    loss = criterion(a_t, gt_t)
    loss.backward()
    sum_grad1 = torch.sum(a_t.grad).item()

    # size is differ
    a_t.grad.data.zero_()
    loss = criterion(a_t.unsqueeze(1), gt_t)  # add dimension
    loss.backward()
    sum_grad2 = torch.sum(a_t.grad).item()
    return sum_grad1, sum_grad2


if __name__ == "__main__":

    np.random.seed(42)

    N = 1000
    bs = 100
    a = np.random.rand(bs, N).astype(np.float64)

    true_index = np.random.randint(N, size=N // 2).reshape(1, N // 2)
    gt = np.zeros((bs, N), dtype=np.float64)
    gt[:, true_index] = 1

    cpu_result = compute_grad(a, gt, device=torch.device("cpu"))
    gpu_result = compute_grad(a, gt, device=torch.device("cuda:0"))

    print("CPU: same shape: {} \t diff shape: {}".format(*cpu_result))
    print("GPU: same shape: {} \t diff shape: {}".format(*gpu_result))

    # CPU: same shape: 1.9732641300852216      diff shape: 1.9732641300852216
    # GPU: same shape: 1.973264130085222       diff shape: 197.3264130085222