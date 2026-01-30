import torchvision.datasets as datasets
input_transform = None

imagenet = datasets.ImageNet("path2imagenet", split="val", transform=input_transform, target_transform=None, download=False)

import torchvision.datasets as datasets
input_transform = None
datasets.imagenet.ARCHIVE_DICT['devkit']['url'] = "https://github.com/goodclass/PythonAI/raw/master/imagenet/ILSVRC2012_devkit_t12.tar.gz"
imagenet = datasets.ImageNet("/dataset-imagenet-ilsvrc2012/", split="val", transform=input_transform, target_transform=None, download=False)