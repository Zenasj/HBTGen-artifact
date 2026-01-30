from multiprocessing import Process
import torch

print(torch.__version__)

def f():
    torch.cuda.set_device(0)

p = Process(target=f)
p.start()
p.join()