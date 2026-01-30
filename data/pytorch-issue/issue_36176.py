import torch
from torch.utils import data

class MapDataset(data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, x):
        return x

class IterableDataset(data.IterableDataset):
    def __len__(self):
        return 10

    def __iter__(self):
        return iter(range(10))


map_loader = data.DataLoader(MapDataset(), batch_size=2)
iterable_loader = data.DataLoader(IterableDataset(), batch_size=2)

print(f"map:  {len(map_loader)}")
print(f"map data: {list(map_loader)}")
print(f"iter: {len(iterable_loader)}")
print(f"iter data: {list(iterable_loader)}")
# map:  5
# map data: [tensor([0, 1]), tensor([2, 3]), tensor([4, 5]), tensor([6, 7]), tensor([8, 9])]
# iter: 10
# iter data: [tensor([0, 1]), tensor([2, 3]), tensor([4, 5]), tensor([6, 7]), tensor([8, 9])]