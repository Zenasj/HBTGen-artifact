import torch

def run(size):
    while True:
        values = torch.zeros(100, 100000)
        target = [torch.zeros(100, 100000)] * size
        dist.all_gather(tensor_list=target, tensor=values)
        time.sleep(1)

del target
gc.collect()