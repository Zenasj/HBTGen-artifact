import torch
n = 0
batches = []
a = random_fullrank_matrix_distinct_singular_value(n, *batches, dtype=torch.float32).to('cpu')
torch.linalg.inv(a)