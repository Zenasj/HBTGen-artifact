import torch

def stack(data, dim=0):
    shape = data[0].shape  # need to handle empty list
    shape = shape[:dim] + (len(data),) + shape[dim:]
    x = torch.cat(data, dim=dim)
    x = x.reshape(shape)
    # need to handle case where dim=-1
    # which is not handled here yet
    # but can be done with transposition
    return x