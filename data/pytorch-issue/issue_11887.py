class DeephomographyDataset(Dataset):
    '''
    DeepHomography Dataset
    '''
    def __init__(self,hdf5file,imgs_key='images',labels_key='labels',
                 transform=None):
        '''
        :argument
        :param hdf5file: the hdf5 file including the images and the label.
        :param transform (callable, optional): Optional transform to be
        applied on a sample
        '''
        self.db=h5py.File(hdf5file,'r') # store the images and the labels
        keys=list(self.db.keys())
        if imgs_key not in keys:
            raise(' the ims_key should not be {}, should be one of {}'
                  .format(imgs_key,keys))
        if labels_key not in keys:
            raise(' the labels_key should not be {}, should be one of {}'
                  .format(labels_key,keys))
        self.imgs_key=imgs_key
        self.labels_key=labels_key
        self.transform=transform
    def __len__(self):
        return len(self.db[self.labels_key])
    def __getitem__(self, idx):
        image=self.db[self.imgs_key][idx]
        label=self.db[self.labels_key][idx]
        sample={'images':image,'labels':label}
        if self.transform:
            sample=self.transform(sample)
        return sample

#   samplesss=trainDataset[0]

#   samplesss=trainDataset[0]