import argparse
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', metavar='int', type=int, dest='rank', default=0, help='rank')
args = parser.parse_args()


torch.cuda.set_device(args.rank)
torch.distributed.init_process_group(backend='nccl')
class test(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(10,10)

    def forward(self, data):
        return self.l(data)

net = nn.parallel.DistributedDataParallel(test(), device_ids=[args.rank], output_device=args.rank)