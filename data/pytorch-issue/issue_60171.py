import torch
from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self):
        super(Mydataset, self).__init__()
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return idx


def main():
    data_set = Mydataset()
    data_set_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=2,
                                                  shuffle=False,
                                                  num_workers=4)
    # refer to
    # https://github.com/pytorch/vision/blob/master/references/detection/engine.py#L72-L109
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    for data in data_set_loader:
        pass

    torch.set_num_threads(n_threads)


if __name__ == '__main__':
    main()