import torch

def pack(x):
    name = os.path.join(tmp_dir, str(uuid.uuid4()))
    torch.save(x, name)
    return name

def unpack(name):
    return torch.load(name)