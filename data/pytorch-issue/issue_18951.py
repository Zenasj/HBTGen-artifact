from __future__ import print_function
import h5py
import numpy as np
import random
import os

if not os.path.exists('./data_h5'):
        os.makedirs('./data_h5')

for index in range(100):
    data = np.random.uniform(0,1, size=(3,128,128))
    data = data[None, ...]
    print (data.shape)
    with h5py.File('./data_h5/' +'%s.h5' % (str(index)), 'w') as f:
        f['data'] = data

import h5py
import torch.utils.data as data
import glob
import torch
import numpy as np
import os
class custom_h5_loader(data.Dataset):

    def __init__(self, root_path):
        self.hdf5_list = [x for x in glob.glob(os.path.join(root_path, '*.h5'))]
        self.data_list = []
        for ind in range (len(self.hdf5_list)):
            self.h5_file = h5py.File(self.hdf5_list[ind])
            data_i = self.h5_file.get('data')     
            self.data_list.append(data_i)

    def __getitem__(self, index):
        self.data = np.asarray(self.data_list[index])   
        return (torch.from_numpy(self.data).float())

    def __len__(self):
        return len(self.hdf5_list)

from dataloader import custom_h5_loader
import torch
import torchvision.datasets as dsets

train_h5_dataset = custom_h5_loader('./data_h5')
h5_loader = torch.utils.data.DataLoader(dataset=train_h5_dataset, batch_size=2, shuffle=True, num_workers=4)      
for epoch in range(100000):
    for i, data in enumerate(h5_loader):       
        print (data.shape)