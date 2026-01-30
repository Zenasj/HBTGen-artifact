import torch
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self,idx):
        return self.get_empty_tensor()

    def get_empty_tensor(self):
        adj = torch.ones([0,2],dtype = torch.long)
        vals = torch.ones([0], dtype = torch.long)
        return {'idx': adj.t(),'vals': vals}

if __name__ ==  '__main__':
    data = dataset()
    dl = DataLoader(data,shuffle = True, num_workers = 1)
    for t in dl:
        print(t)