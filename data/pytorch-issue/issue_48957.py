from torch.utils.data import DataLoader
import numpy as np
from data import Dataset, set_var

if __name__ == '__main__':
    set_var()
    test_data=Dataset()

    print("DataLoader with num_workers=0 (sequential):")
    data_loader=DataLoader(test_data)
    for batch in data_loader:
        pass
        
    print("DataLoader with num_workers=2 (parallel):")
    data_loader=DataLoader(test_data,num_workers=2)
    for batch in data_loader:
        pass

import torch

VAR=0
print("Initialized VAR to",VAR)

def set_var():
    global VAR
    VAR=1
    print("Setting VAR to",VAR)

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dummy dataset for PyTorch'
  def __init__(self):
        'Initialization'

        self.x_data=torch.tensor([[0,1],[2,3]])
        self.y_data=torch.tensor([4,5])

  def __len__(self):
        'Denotes the total number of samples'
        
        return len(self.x_data)

  def __getitem__(self, index):
        'Returns one sample of data'
        
        X =self.x_data[index]
        y=self.y_data[index]

        print("... In Dataset.__getitem__, VAR is currently",VAR,"(should be 1)")

        return X, y