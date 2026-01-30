import torch

class LXRTDataLoader(torch.utils.data.Dataset):
    def __init__(self):
        """do not open hdf5 here!!"""

    def open_hdf5(self):
        self.img_hdf5 = h5py.File('img.hdf5', 'r')
        self.dataset = self.img_hdf5['dataset'] # if you want dataset.

    def __getitem__(self, item: int):
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
        img0 = self.img_hdf5['dataset'][0] # Do loading here
        img1 = self.dataset[1]
        return img0, img1

train_loader = torch.utils.data.DataLoader(
        dataset=train_tset,
        batch_size=32,
        num_workers=4
    )

def __del__(self):
    if hasattr(self, 'img_hdf5'):
        self.img_hdf5.close()

def __init__(self):
    with h5py.File("X.hdf5", 'r') as f:
        self.length = len(f['dataset'])

def __len__(self):
    return self.length