import torch
torch.multiprocessing.set_start_method("spawn")
import torch.multiprocessing as mp


def sub_processes(A, B, D, i, j, size):

    D[(j * size):((j + 1) * size), i] = torch.mul(B[:, i], A[j, i])


def task(A, B):
    size1 = A.shape
    size2 = B.shape
    D = torch.zeros([size1[0] * size2[0], size1[1]]).cuda()
    D.share_memory_()

    for i in range(1):
        processes = []
        for j in range(size1[0]):
            p = mp.Process(target=sub_processes, args=(A, B, D, i, j, size2[0]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    return D

A = torch.rand(3, 3).cuda()
B = torch.rand(3, 3).cuda()
C = task(A,B)
print(C)

import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

ctx = multiprocessing.get_context("spawn")

print(torch.multiprocessing.get_start_method())

import datasets
wnut = datasets.load_dataset("wnut_17")
import torch 
torch.multiprocessing.set_start_method('spawn')

torch.multiprocessing.set_start_method("spawn", force=True)