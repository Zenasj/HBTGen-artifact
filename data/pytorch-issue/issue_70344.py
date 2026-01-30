import torch
from torchvision import transforms
import torchvision
import torch.utils.data

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=True
)

batch_size = 256

def get_dataloader_workers():
    return 4

train_iter = torch.utils.data.DataLoader(
    mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()
)

for X, y in train_iter:
    continue