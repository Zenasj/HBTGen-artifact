import torch
import numpy as np

device = torch.device('cuda:0')


class TrainDataset(Dataset):
    def __init__(self, data, target_cols):
        self.X = np.array(data.drop(target_cols, axis=1))
        self.y = np.array(data[target_cols])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        input = torch.from_numpy(self.X[idx]).float()
        target = torch.from_numpy(self.y[idx]).float()
        return input.to(device), target.to(device)

X_train_set = TrainDataset(X_train, target_cols=['CONDOMINIUM_EXPENSES'])
X_train_loader = DataLoader(X_train_set, batch_size=4, shuffle=True, num_workers=4)