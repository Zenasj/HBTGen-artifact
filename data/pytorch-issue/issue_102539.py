import torch.nn as nn

import torch
import torch._dynamo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

def norm_img(x):
    return (x - x.min()) / (x.max() - x.min())

class TrainMIT5KDataset(Dataset):
    def __init__(self, datadir):
        self.datadir = Path(datadir)
        image_paths = self.datadir / 'train_processed' / 'raw'
        target_paths = self.datadir / 'train_processed' / 'target'
        self.image_paths = sorted(image_paths.glob('*.jpg'))
        self.target_paths = sorted(target_paths.glob('*.jpg'))
        assert [ipath.name for ipath in self.image_paths] == [tpath.name for tpath in self.target_paths]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target_path = self.target_paths[index]
        image = norm_img(to_tensor(Image.open(image_path)))
        target = to_tensor(Image.open(target_path)) 
        return image, target

    def __len__(self):
        return len(self.image_paths)




def fit(enhancer, dataloader, optimizer, scheduler, loss_fn, n_epochs=24, verbose=True, profiler=None):
    logger = SummaryWriter()

    for epoch_idx in range(n_epochs):
        for i, (raw_batch, target_batch) in enumerate(dataloader):
            print('epoch', epoch_idx, 'batch', i)
            out_batch = enhancer(raw_batch)
            target_batch = target_batch.float().reshape(target_batch.shape[0], -1).mean(dim=1, keepdim=True)
            loss = loss_fn(out_batch, target_batch)
            # loss = loss_fn(out_batch, target_batch[...,None].float())
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            logger.add_scalar("train/loss", loss, epoch_idx * len(dataloader) + i)

            if profiler:
                # save profiler stats
                profiler.disable()
                profiler.dump_stats(f"tests/profiler.prof")
                profiler.enable()
    return enhancer





if __name__ == '__main__':
    import cProfile
    # initialize profiler
    pr = cProfile.Profile()
    pr.enable()

    n_params = 1
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Linear(512, n_params)
    backbone = resnet
    enhancer = backbone

    torch._dynamo.config.verbose = True
    enhancer = torch.compile(enhancer, mode='reduce-overhead')
    print(enhancer(torch.zeros(1, 3, 448, 448)))

    lr=1e-3
    optimizer = torch.optim.Adam(enhancer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=50, verbose=True
    )

    dataset = TrainMIT5KDataset(datadir='dataset/C')
    # dataset = CIFAR10(root='dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    loss_fn = torch.nn.MSELoss()
    enhancer = fit(enhancer, dataloader, optimizer, scheduler, loss_fn, n_epochs=24, verbose=True, profiler=pr)

    pr.disable()
    pr.dump_stats(f"tests/profiler.prof")