from torch.utils.data import Dataset
class testSet(Dataset):
    def __init__(self):
        super(testSet,self).__init__()
    def __len__(self):
        return 1000000
    def __getitem__(self,index):
        return {"index": index}

import torch

test_data = testSet()
test_data_loader = torch.utils.data.DataLoader( dataset=test_data, batch_size=1, num_workers=1)
index = []
for sample in test_data_loader:
    index.append(sample['index'])