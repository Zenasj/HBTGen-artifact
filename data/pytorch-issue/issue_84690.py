import torch

def set_gpu(x: str):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(x)
    print('using gpu:', x)

def set_gpu(x: str):
    torch.cuda.set_device(int(x))