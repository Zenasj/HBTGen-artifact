import torch.nn as nn

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import pandas as pd


def calc_cos_sims(rank, world_size):
    group = dist.new_group([0, 2])
    cuda_device = torch.device('cuda:'+str(rank))

    data_path = './embed_pairs_df_8000_part' + str(rank) + '.pkl'
    tmp_df = pd.read_pickle(data_path)
    embeds_a_list = [embed_a for embed_a in tmp_df['embeds_a']]
    embeds_b_list = [embed_b for embed_b in tmp_df['embeds_b']]

    embeds_a_tensor = torch.tensor(embeds_a_list, device=cuda_device)
    embeds_b_tensor = torch.tensor(embeds_b_list, device=cuda_device)

    cosine_tensor = F.cosine_similarity(embeds_a_tensor, embeds_b_tensor)
    cosine_tensors_concat = dist.gather(cosine_tensor, group=group)


def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def main():
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, calc_cos_sims))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



if __name__ == '__main__':
    main()
    print('DONE!')