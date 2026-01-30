from torchvision import datasets, transforms
train_set = datasets.ImageNet('./data',train=True,download=True,
                                 transform = transforms.Compose([transforms.ToTensor()]))

test_set = datasets.ImageNet('./data',train=False,download=True,
                                 transform = transforms.Compose([transforms.ToTensor()]))