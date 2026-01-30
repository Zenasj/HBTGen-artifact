import torch
from torch.utils.data import Dataset, DataLoader


def create_sparse_tensor():
    tensor = torch.randn(5, 5)
    sparse_tensor = tensor.to_sparse().to("cpu")
    torch.save(sparse_tensor, "sparse_tensor.pth")


class OperatorDataset(Dataset):
    def __init__(self):
        self.files = ["sparse_tensor.pth"]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        _ = torch.load(self.files[idx], weights_only=True, map_location="cpu")
        return None


if __name__ == '__main__':
    print(torch.__version__)

    create_sparse_tensor()
    
    dataset = OperatorDataset()
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )
    
    for sparse_tensor in dataloader:
        # Error raised here
        pass

...
if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    ...

# File: test_fork.py
import torch
import os
t = torch.tensor([1, 2])
print(f'1: {t.is_pinned()}')
os.fork()
print(f'2: {t.is_pinned()}')

import torch
import os
t = torch.tensor([1, 2])
# print(f'1: {t.is_pinned()}')
os.fork()
print(f'2: {t.is_pinned()}')