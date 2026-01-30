import multiprocessing as mp
import torch


def train():
    print("Training started")
    x = torch.Tensor(0)
    x.requires_grad = True
    x.sum().backward()
    print("Training ended")


if __name__ == "__main__":
    print("Entered main")
    train()
    worker = mp.Process(target=train)
    worker.start()