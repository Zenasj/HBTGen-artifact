import torch.nn as nn

import torch
import torch.nn.functional as F
from multiprocessing import Pool
import torch.multiprocessing as mp
import os, time, random


def worker(i, norm_feat, all_feat):
    print(f"{i} start")
    sim = torch.mm(norm_feat[i][None, :], norm_feat.t()).squeeze()
    init_rank = torch.topk(sim, 5)[1]
    weights = sim[init_rank].view(-1, 1)
    weights = torch.pow(weights, 2)
    ret = torch.mean(all_feat[init_rank, :] * weights, dim=0)
    print(f"{i} end")

    return ret


if __name__ == "__main__":
    multi_p = 1
    torch_pool = 1
    cuda = 0

    t_start = time.time()

    all_feat = torch.randn(1000, 2048)
    if cuda: all_feat = all_feat.cuda()
    norm_feat = F.normalize(all_feat, p=2, dim=1)
    # all_feat.share_memory_()
    # norm_feat.share_memory_()

    num = len(norm_feat)

    if multi_p:
        if torch_pool:
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(8)
        else:
            pool = Pool(8)

    pool_list = []
    for i in range(num):
        if multi_p:
            res = pool.apply_async(worker, args=(i, norm_feat, all_feat,))
        else:
            res = worker(i, norm_feat, all_feat)

        pool_list.append(res)

    if multi_p:
        pool.close()
        pool.join()

    # print(pool_list)

    cost = time.time()-t_start
    print(f"Cost: {cost:.2f}")