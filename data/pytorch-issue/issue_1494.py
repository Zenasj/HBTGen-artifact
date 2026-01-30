import random

python
# from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

# Above are NOT executed

import torch
from torch.multiprocessing import Process

def boom():
    b = torch.FloatTensor([1,2,3])
    # b = b.cuda(0) # Normal error msg.
    # b = b.cuda(1) # Normal error msg.
    b = torch.autograd.Variable(b)
    # b = b.cuda(1) # Normal error msg.
    b = b.cuda(0) # initialization error at /b/wheel/pytorch-src/torch/lib/THC/generic/THCStorage.c:55

a = torch.FloatTensor([2,2,3])
a = a.cuda(1)
a = torch.autograd.Variable(a)
p = Process(target=boom,args=())
p.daemon = True
p.start()
p.join()

Python 
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import argparse
import torch
from multiprocessing import Process

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-id", default=0, type=int,
                        choices=list(range(torch.cuda.device_count())))

    return parser.parse_args()


def run_in_process():
    print(torch.cuda.device_count())

def main():
    args = parse_args()
    p = Process(target=run_in_process)
    p.start()
    p.join()

if __name__ == '__main__':
    main()

class DummyDataset(data.Dataset):
    def __init__(self, num_classes):
        super(DummyDataset, self).__init__()
        # Create tensor on GPU 
        self.tensor = torch.randn(3, 224, 224, device=torch.device('cuda'))
        self.num_classes = num_classes

    def __getitem__(self, index):
        torch.manual_seed(index)
        random.seed(index)
        return self.tensor, \
               random.randint(0, self.num_classes - 1)