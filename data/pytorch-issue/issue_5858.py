import torch

torch.multiprocessing.freeze_support()

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()