import multiprocessing as mp
import torch
import time

q = mp.Queue()

def sender():
    global q
    q.put(torch.zeros(20, 20, 20))
   # time.sleep(20.0)


pp = mp.Process(target=sender, args=())
pp.start()
res = q.get()
print("Got tensor with mean", res.mean())

import torch
import torch.utils.data
import torchvision
import time

torch.cuda.set_device(0)

dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
            "~/datasets",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    ),
    pin_memory=True,
    batch_size=1024,
    persistent_workers=True,
    num_workers=8,
)

for _ in dataloader:
    break

# print("Sleeping for 5 seconds"); time.sleep(5) # Uncomment this line to make the error disappear

print("Finished!")