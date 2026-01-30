import torch

def printShape(shm):
    tensor = torch.frombuffer(shm.buf, dtype = torch.float32)
    print(len(shm.buf))
    print(tensor)
    print(tensor.shape)