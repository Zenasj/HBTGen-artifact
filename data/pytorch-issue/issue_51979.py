py
import torch.utils.data

class MyException(Exception):
    def __init__(self, msg, arg1, arg2):
        super().__init__(msg)

class MyDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        raise MyException("MyError", 1, 2)

if __name__ == '__main__':
    dl = torch.utils.data.DataLoader(MyDataset(), num_workers=1)
    list(iter(dl))