import torch
import torch.nn as nn
import numpy as np
import random

class SiameseDataset(Dataset):
    def __init__(self, root_dir):
        super(SiameseDataset, self).__init__()
        self.h_table = {}
        self.root_dir = root_dir
        self.folders = os.listdir(root_dir)
        self.folders.sort()
        self.num_folders = len(self.folders) - 1
        for folder in self.folders:
            self.h_table[str(int(folder) - 1)] = os.listdir("{}/{}".format(root_dir, folder))

    def __len__(self):
        return len(self.h_table)

    def __getitem__(self, index):
        same = random.uniform(0,1) > 0.5
        h_len = len(self.h_table[str(index)]) - 1
        h_ = self.h_table[str(index)]
        if same:
            first_idx = random.randint(0, h_len)
            while True:
                second_idx = random.randint(0, h_len)
                if first_idx != second_idx:
                    break

            first_path = "{}/{:04d}/{}".format(self.root_dir, index + 1, h_[first_idx])
            second_path = "{}/{:04d}/{}".format(self.root_dir, index + 1, h_[second_idx])

            first = np.array(plt.imread(first_path))
            second = np.array(plt.imread(second_path))

            first = transforms.ToTensor()(first)
            second = transforms.ToTensor()(second)
            
            return (first, index), (second, index)
        else:
            first_idx = random.randint(0, h_len)
            while True:
                sec_class_idx = random.randint(0, self.num_folders)
                if sec_class_idx != index:
                    break
            second_idx = random.randint(0, h_len)
            second = self.h_table[str(sec_class_idx)][second_idx]

            first_path = "{}/{:04d}/{}".format(self.root_dir, index + 1, h_[first_idx])
            second_path = "{}/{:04d}/{}".format(self.root_dir, index + 1, h_[second_idx])

            first = np.array(plt.imread(first_path))
            second = np.array(plt.imread(second_path))

            first = transforms.ToTensor()(first)
            second = transforms.ToTensor()(second)

            return (first, index), (second, sec_class_idx)

torch.multiprocessing.set_start_method('spawn')
device = torch.device("cuda")

root_dir = "data/256x256/1"
dataset = SiameseDataset(root_dir)
batch_size = 128 * torch.cuda.device_count()
loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

model = SiameseNeuralNetwork().to(device)
model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
learning_rate = 1e-3
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
num_epochs = 10