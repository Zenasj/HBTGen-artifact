import torchvision.datasets as datasets
input_transform = None

imagenet = datasets.ImageNet("path2imagenet", split="val", transform=input_transform, target_transform=None, download=False)