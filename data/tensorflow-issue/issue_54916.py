import numpy as np

class My_Custom_Generator(Sq):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array([
            resize(imread('E:\\data\\allFrames\\' + str(file_name)), (559, 331))
            for file_name in batch_x]) / 255.0, np.array(batch_y)