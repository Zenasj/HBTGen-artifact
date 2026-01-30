import random

import torch
import sys
import  pandas as pd
import numpy as np


def get_mat_dist_pow():
    p10 = 0.7 # a matrix_size of 10 is highly likely to fail, so increase probability
    p_other = (1 - p10) / 46
    dist_matrix_size = {v.long().item(): p_other for v in torch.linspace(4, 50, 47)}
    dist_matrix_size[10] = p10
    return dist_matrix_size


def get_batch_dist_pow():
    return {2**e.long().item(): 1/11. for e in torch.linspace(0, 10, 11)}


def get_matrix_size():
    # https://github.com/pytorch/pytorch/blob/47f0bda3ef8d196e0fa81a1749814dd75ffb1692/torch/utils/benchmark/utils/fuzzer.py#L120
    index = state.choice(
        np.arange(len(mat_dist_pow)),
        p=tuple(mat_dist_pow.values()))
    matrix_size = list(mat_dist_pow.keys())[index]
    return matrix_size


def get_batch_size(p_switch=0.5):
    # switch between uniform and pow
    if torch.empty(1).uniform_() > p_switch:
        batch_size = torch.empty(1).uniform_(1, 1024*1024).long()
    else:
        # https://github.com/pytorch/pytorch/blob/47f0bda3ef8d196e0fa81a1749814dd75ffb1692/torch/utils/benchmark/utils/fuzzer.py#L120
        index = state.choice(
            np.arange(len(batch_dist_pow)),
            p=tuple(batch_dist_pow.values()))
        batch_size = list(batch_dist_pow.keys())[index]
    return batch_size


dtype = torch.float32
nb_iters = 600000
columns=["matrix_size", "batch_size", "passed"]
result = pd.DataFrame(columns=columns)
device = 'cuda:0'
mat_dist_pow = get_mat_dist_pow()
batch_dist_pow = get_batch_dist_pow()
state = np.random.RandomState(2809)

for _ in range(nb_iters):
    try:
        # create fake data on device to move workload
        size = 2**torch.randint(0, 30, (1,))
        fake = torch.randn(size, device=device)
        print('Created fake tensor of {}MB'.format(fake.nelement()*4/1024**2))

        matrix_size = get_matrix_size()
        batch_size = get_batch_size()
        print('Using matrix_Size {}, batch_size {}'.format(matrix_size, batch_size))
        input = torch.eye(matrix_size, device=device, dtype=dtype).expand(batch_size, -1, -1)
        print('input.shape {}'.format(input.shape))

        # execute test multiple times
        for _ in range(3):
            ret = torch.cholesky(input)
        torch.cuda.synchronize()
        passed = torch.isfinite(ret).all()
        if not passed:
            print('ERROR! Invalid values found!')
            print(ret)
            torch.save(ret, 'invalid_ret01.pt')
            sys.exit(-1)

        # save results
        result = result.append(pd.DataFrame([[matrix_size, batch_size, passed.item()]], columns=columns))
        del fake
        del input
        del ret
    except Exception as e:
        print(e)
        if 'out of memory' in str(e):
            result = result.append(pd.DataFrame([[matrix_size, batch_size, 'oom']], columns=columns))
            continue
        else:
            result = result.append(pd.DataFrame([[matrix_size, batch_size, str(e)]], columns=columns))
            continue

result.to_csv('magma_ima_result01.csv', index=False)