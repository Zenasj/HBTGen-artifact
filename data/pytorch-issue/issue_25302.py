import numpy as np
import torch 

class MazeDataset(torch.utils.data.Dataset):
    def __init__(self):
      self.data = np.arange(40)

    def __getitem__(self, idx):   
        video, target = torch.as_tensor(self.data[idx], dtype=torch.float32), torch.as_tensor(5, dtype=torch.float32)
        return video, target
    
    def __len__(self):
        return 40


if __name__ == '__main__':
    dataset = MazeDataset()

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=True, pin_memory= True, num_workers = 8)

    for input, target in data_loader:
        print(target)