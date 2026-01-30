import torch
import numpy as np
import random

class HuBMAPDataset(Dataset):
    def __init__(self, image_path='train_image_tiles/', mask_path='train_mask_tiles/', 
                 tfm=None, mean=None, std=None):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.nrm = transforms.Normalize(mean=mean, std=std)
        self.tfm = tfm
        self.image_list = os.listdir(self.image_path)

        assert len(self.image_list) == len(os.listdir(self.mask_path))
        self.mask_list = [x[:-5]+'.npy' for x in self.image_list]


    def __getitem__(self, idx):
        image = io.imread(self.image_path + self.image_list[idx])
        # adding .astype('int32') to the next file fixes the issue!!!!!!!!!!!
        mask = np.load(self.mask_path + self.mask_list[idx])
        print(mask.dtype)

        seed = np.random.randint(81261917)
        torch.manual_seed(seed) # needed for torchvision
        image = self.tfm(image)
        image = self.nrm(image)

        torch.manual_seed(seed) # needed for torchvision
        mask = self.tfm(mask)
    
        return image, mask

    def __len__(self):
        return len(self.image_list)


mean = [0.6214918394883474, 0.522968004147211, 0.6425345639387766]
std = [0.2847020526727040, 0.312313973903656, 0.2859026342630386]

tfm = transforms.Compose([transforms.ToTensor(), 
                          transforms.RandomRotation(90, expand=False, center=None, fill=None), 
                          transforms.RandomVerticalFlip(p=0.5), 
                          transforms.RandomHorizontalFlip(p=0.5)])

dataset = HuBMAPDataset('train_image_tiles/', 'train_mask_tiles/', tfm=tfm, mean=mean, std=std)
n_samples = len(dataset)
batch_size = 4
n_val = int(0.1*n_samples)
n_train = n_samples - n_val
n_workers = 0
n_channels = 3

train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=n_workers, pin_memory=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, 
                        num_workers=n_workers, pin_memory=False, drop_last=True)

mask = torch.as_tensor(np.array(mask), dtype=torch.int64)