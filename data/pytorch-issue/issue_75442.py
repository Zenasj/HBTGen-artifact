import torch

from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(self):
        pass
    def __getitem(self,index):
        pass
        return {'img': Array, 'txt': str}

inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)

def forward(self, input):
        idx = torch.cuda.current_device()
        count = torch.cuda.device_count()
        txt = input['txt']
        txt = txt[idx * len(txt) : len(txt) // count]