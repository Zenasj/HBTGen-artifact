import torch
from torch.utils._benchmark import Timer

population_size = 10000
max_score = 10000
device = torch.device("cuda:0")
s1, s2 = [], []

def first():
    for _ in range(1000):
        a = torch.randint(max_score, size=(100 * population_size,))  # selection,mutation create x100 instance.
        probs = a.float()
        probs = probs / probs.sum(-1, keepdim=True)
        s1.append(probs)

def second():
    for _ in range(1000):
        a = torch.randint(max_score, size=(100 * population_size,))  # selection,mutation create x100 instance.
        probs = a.true_divide(a.sum(-1, keepdim=True))
        s2.append(probs)

start = Timer('lambda: first()')
print(torch.__version__)
print(start.timeit())
start = Timer('lambda: second()')
print(start.timeit())