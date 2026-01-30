import torch
from tqdm import trange

DTYPE = torch.float32
MAT_SIZE = 5000
DEVICE = ["cpu", "mps"][0]      # it's CPU now

mat = torch.randn([MAT_SIZE, MAT_SIZE], dtype=DTYPE, device=DEVICE)

for i in trange(N_ITER := 100):
    mat @= mat                  # <--- Main Computation HERE
    print(mat[0, 0], end="")    # avoid sync-issue when using 'mps'