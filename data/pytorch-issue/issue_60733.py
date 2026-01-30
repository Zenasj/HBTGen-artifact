import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import types
import argparse


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 10, bias=False)

    def __init_opt(self):
        param = next(self.parameters())
        opt = torch.randn(1, 10, device=param.device)
        return opt

    def forward(self, x, opt):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if opt is None:
            opt = self.__init_opt()
        return x, opt


def main():
    parser = argparse.ArgumentParser(description='fdsa')
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    model = MyModel().to(args.gpu)
    model = DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False
    )

    opt = [None]
    for i in range(2):
        model.zero_grad()
        x = torch.randn(1, 10, device=args.gpu)
        out, opt[0] = model(x, opt=opt[0])
        print('iter {}, out.sum() {}, device {}, opt.grad_fn {}, opt.device {}'.format(
            i, out.sum(), out.device, opt[0].grad_fn, opt[0].device))
        out.mean().backward()

if __name__ == "__main__":
    main()