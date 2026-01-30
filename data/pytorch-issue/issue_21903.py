import torch
import torchvision
import torchvision.transforms as transforms

print("Hello World!")

def main():
    transform = transforms.ToTensor() 
    trainset = torchvision.datasets.MNIST(root='./data/', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                     shuffle=True, num_workers=4)
    for i, data in enumerate(trainloader):
        pass

if __name__ == '__main__':
    main()