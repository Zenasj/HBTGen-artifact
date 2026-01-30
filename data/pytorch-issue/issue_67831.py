import torch

from torch.utils.data import DataLoader

class CustomDictClass(dict):
    def get_something(self):
        return "something"

dataset = [{"values": [0, 1, 2]}, {"values": [2, 3, 1]}]

def collate_fn(batch):
    return CustomDictClass({key: [elem[key] for elem in batch] for key in batch[0].keys()})

dataloader = DataLoader(
    dataset,
    pin_memory=True,
    collate_fn=collate_fn,
)

for batch in dataloader:
    print(batch)
    print(type(batch))
    print(batch.get_something())