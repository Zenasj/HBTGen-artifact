import torch.nn as nn

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 5
rows = 10000
cols = 1000
trails = 10
num_samples = 1
for i in range(trails):
    torch.manual_seed(seed)
    probs = torch.rand(rows, cols, dtype=torch.float16, device="cuda")
    fake_token1 = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    torch.manual_seed(seed)
    probs = torch.rand(rows, cols, dtype=torch.float16, device="cuda")
    fake_token2 = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    diff_ratio = (fake_token1 != fake_token2).sum().item() / probs.shape[0] / num_samples
    print('trail {} inconsistency ratio = {}'.format(i, diff_ratio))
    if diff_ratio > 0:
        print(torch.nonzero(fake_token1 != fake_token2, as_tuple=False))
        print(fake_token1[fake_token1 != fake_token2])
        print(fake_token2[fake_token1 != fake_token2])