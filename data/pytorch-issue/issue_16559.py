for i in range(num_gpu):
    with torch.cuda.device(i):
        torch.tensor([1.]).cuda()

import torch
import threading

def worker(rank):
    torch.tensor([1.]).cuda(rank)

# to fix the isuue, uncomment the following
# torch.tensor([1.]).cuda(0)
# torch.tensor([1.]).cuda(1)


t1 = threading.Thread(target=worker, args=(0,))
t2 = threading.Thread(target=worker, args=(1,))
t1.start()
t2.start()

def __load__(self, args):
        (file, idx, out_size) = args
        image = Image.open(file).convert('RGB')
        t_out       = torch.zeros((out_size,), dtype=torch.long)
        t_out[idx]  = 1
        t_in        = self.tf(image)
        return (t_in.cuda().float(), t_out.cuda().long())

use_cuda = torch.cuda.is_available()
assert use_cuda == True
device = torch.device("cuda:0")