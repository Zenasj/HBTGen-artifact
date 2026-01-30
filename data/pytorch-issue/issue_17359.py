import torch
import torch.utils.data
from collections import defaultdict

class Dataset(torch.utils.data.Dataset):
  
  def __init__(self):
    self.data = defaultdict(dict)
    for i in range(32):
      self.data[i]['sequence'] = list(range(10))
    
  def __getitem__(self, idx):
    return self.data[idx]
 
  def __len__(self):
    return len(self.data)
  

def collate_fn(data):
  batch = defaultdict(list)
  for key in data[0].keys():
    batch[key] = [s[key] for s in data]
    batch[key] = torch.Tensor(batch[key]).long()
    batch[key] = batch[key].to(torch.device('cuda'))
      
  return batch

# default collate works
try:
  dataloader = torch.utils.data.DataLoader(Dataset(), 4, num_workers=2)
  for epoch in range(2):
    for iteration, batch in enumerate(dataloader):
      pass
  print("Default collate_fn works!")
except Exception as e:
  print("Default collate_fn failed!")
  print(e)

# custom collate fails
try:
  dataloader = torch.utils.data.DataLoader(Dataset(), 4, num_workers=2, collate_fn=collate_fn)
  for epoch in range(2):
    for iteration, batch in enumerate(dataloader):
      print(epoch, iteration)
  print("Custom collate_fn works!")
except Exception as e:
  print("Custom collate_fn failed!")
  print(e)