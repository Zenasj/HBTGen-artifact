import torch

def aa():
    print("==== here is aa ====")
    print(torch)


def bb():
    global torch
    print("==== here is bb ====")
    print(torch)
    import torch.profiler

aa()
bb()