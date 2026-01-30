import torchvision.datasets as dset
class ImageFolderEX(dset.ImageFolder):
    def __getitem__(self, index):
        path, label = self.imgs[index]
        try:
            img = self.loader(os.path.join(self.root, path))
        except:
            pass #your handling code
        return [img, label]

import nonechucks as nc

dataset = ImageFolder('...')
dataset = nc.SafeDataset(dataset)

class ImageFolderEX(dset.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except:
            return None
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target