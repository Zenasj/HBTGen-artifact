import torch.utils.data
class E(Exception):
    def __init__(self, a, b):
        pass
    def __str__(self):
        return "Special exception message"

class G(torch.utils.data.IterableDataset):
    def __iter__(self):
        raise E(a=1, b=2)
        yield 1
    
d = torch.utils.data.DataLoader(G(), num_workers=1)
next(iter(d))