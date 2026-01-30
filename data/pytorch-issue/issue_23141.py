import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "foo": np.array([1, 2, 3]),
            "bar": ["X"] * (idx+1),
        }

training = CustomDataset()

for batch in DataLoader(training, batch_size=2):
    print(batch)

{
  'foo': tensor(
    [
      [1, 2, 3],
      [1, 2, 3]
    ]
  ),
  'bar': [
      ('X', 'X'),
    ]
}

{
  'foo': tensor(
    [
      [1, 2, 3],
      [1, 2, 3]
    ]
  ),
  'bar': [
      ('X'),
      ('X', 'X'),
    ]
}