import torch
import numpy as np
import random

def cholesky_speed_test(device="cpu"):
    assert device in ("cpu","cuda")
    np.random.seed(123)
    n_rep = 10000
    dim = 16
    batches = 10

    A = np.random.normal(0,1,(batches,dim,dim))
    cov = np.matmul(A,A.swapaxes(1,2))

    pt_cov = torch.tensor(cov.astype(np.float32),device=device)
    L = torch.cholesky(pt_cov)

    torch.cuda.nvtx.range_push("CholeskyDecomposition")
    start_time = time.time()
    for i in range(n_rep):
        torch.cuda.nvtx.range_push("Iter{}".format(i))
        torch.cholesky(pt_cov, out=L)
        torch.cuda.nvtx.range_pop()
    duration = time.time() - start_time
    print("Duration: {:.3f}s on device {}".format(duration,device))
    torch.cuda.nvtx.range_pop()