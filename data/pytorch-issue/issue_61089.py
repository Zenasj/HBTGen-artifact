import torchvision

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, TensorDataset
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307, ),
        (0.3081, )),  # normalize should be executed after ToTensor()
])


def load_img(x):
    return Image.open(x)


label_set = DatasetFolder("./data/training/labeled",
                          loader=load_img,
                          extensions="jpg",
                          transform=train_tfm)

test_img = torch.ones(1000, 3, 128, 128)
test_label = torch.ones(1000)
test_set = TensorDataset(test_img, test_label)

concat_set = ConcatDataset([label_set, test_set])

train_loader = DataLoader(concat_set, batch_size=128, shuffle=True)
for batch in train_loader:
    img, label = batch
    print(label.shape)