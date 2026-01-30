import argparse                                                                                                                                                                                                                           
import os
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
 
 
class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
 
    def forward(self, x):
        x = self.conv1(x)
        return x
 
 
def Main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
 
    torch.distributed.init_process_group(backend='nccl')
 
    model = ToyNet()
 
    device = torch.device('cuda', args.local_rank)
    model = model.to(device)
    distrib_model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
 
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    eval_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
 
    sampler = DistributedSampler(eval_set)
    dataloader = DataLoader(dataset=eval_set, batch_size=16, sampler=sampler)
 
    distrib_model.eval()
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        with torch.no_grad():
            predictions = distrib_model(inputs.to(device))         # Forward pass
        print('batch_idx: {}'.format(batch_idx))
 
 
if __name__ == "__main__":
    Main()