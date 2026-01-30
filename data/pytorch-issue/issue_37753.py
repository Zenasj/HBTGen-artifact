import torch
import torch.utils.data as data

class MyDataset(object):
    def __init__(self):
        super().__init__()
        self.n = 1000000
    def __len__(self):
        return self.n
    def __getitem__(self, index):
        return torch.zeros((99, 99)), torch.zeros((99, 99))
            
dataset = MyDataset()
loader = data.DataLoader(dataset, batch_size=128)
print(len(dataset), len(loader))

class MyDataset2(data.IterableDataset):
    def __init__(self):
        super().__init__()
        self.n = 1000000
    def __len__(self):
        return self.n
    def __iter__(self):
        for i in range(self.n):
            yield torch.zeros((99, 99)), torch.zeros((99, 99))
            
dataset2 = MyDataset2()
loader = data.DataLoader(dataset2, batch_size=128)
print(len(dataset), len(loader))