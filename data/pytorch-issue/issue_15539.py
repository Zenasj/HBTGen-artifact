import torch

def main():
    def test_cholesky():
        A = torch.rand([4, 1280, 1280]).cuda()
        H = A.matmul(A.transpose(1, 2)) + torch.eye(1280).cuda()
        decomposition = torch.cholesky(H, upper=False)
        b = torch.rand([4, 1280]).cuda()
        return torch.potrs(b.unsqueeze(2), decomposition, upper=False).squeeze(2), decomposition

    for i in range(10000):
        test_cholesky()
        print(i)


if __name__ == '__main__':
    main()

import torch
import gc
import resource

def test_cholesky():
    A = torch.rand(4, 1280, 1280).cuda()
    H = A.matmul(A.transpose(1, 2))
    decomp = torch.cholesky(H)

if __name__ == '__main__':
    for i in range(0, 100):
        gc.collect()
        test_cholesky()
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)