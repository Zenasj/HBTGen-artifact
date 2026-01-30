import torch
import torchvision

file = open('test.txt', 'w')

if __name__ == "__main__":
    file.write("test1\n")
    file.flush()
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=torchvision.transforms.ToTensor()),
        batch_size=128, shuffle=True, num_workers=2)
    for _ in enumerate(train_loader):
        pass
    file.write("test2\n")
    file.flush()

test1
test2

test2