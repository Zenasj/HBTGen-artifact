import torch

x = [[[3, 2, 3],[1, 3, 4]],
       [[3, 1, 2],[3, 2, 4]],
       [[4, 4, 2], [1, 1, 1]]] 
torch.cuda.comm.scatter(torch.tensor(x), [1], chunk_sizes=None, dim=0, streams=[1, 2])