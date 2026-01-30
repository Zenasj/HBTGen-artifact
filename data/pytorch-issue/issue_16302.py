import torch

q = torch.multiprocessing.Queue()
while True:
    t = torch.ones(1)
    q.put(t)
    q.get()