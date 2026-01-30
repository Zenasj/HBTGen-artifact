import torch
import torch.nn as nn
import torch.nn.functional as F


def unfold1(x, chunk_size):
    hop_size = chunk_size // 2
    x = x.transpose(1, 2)
    x = x.unfold(-1, chunk_size, hop_size)
    return x


def unfold2(x, chunk_size):
    hop_size = chunk_size // 2
    x = x.transpose(1, 2)
    B, C, T = x.shape
    x = x.reshape(B, C, T // hop_size, hop_size)
    x = torch.cat((x[:, :, :-1], x[:, :, 1:]), dim=-1)
    return x


if __name__ == '__main__':
    device = 'cuda'
    x = torch.arange(24).reshape(2, 6, 2).float().to(device)
    print(unfold1(x, 4))
    print('-' * 50)
    print(unfold2(x, 4))