import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

if __name__=="__main__":
    device = torch.device("mps")
    dataset = MNIST(root='data/', download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    real_batch = next(iter(dataloader))
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)).cpu(),(1,2,0)))
    plt.show()

def repro():
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    device = torch.device("mps")
    dataset = MNIST(root='data/', download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    real_batch = next(iter(dataloader))
    plt.imshow(torch.permute(vutils.make_grid(real_batch[0].to(device)).cpu(),(1,2,0)))
    plt.show()