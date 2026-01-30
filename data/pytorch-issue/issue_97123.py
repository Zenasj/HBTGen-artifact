from mmengine.dist import all_gather, broadcast, get_rank, init_dist

import torch


def batch_shuffle_ddp(x: torch.Tensor):
    """Batch shuffle, for making use of BatchNorm.
    Args:
        x (torch.Tensor): Data in each GPU.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Output of shuffle operation.
            - x_gather[idx_this]: Shuffled data.
            - idx_unshuffle: Index for restoring.
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = torch.cat(all_gather(x), dim=0)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all)

    # broadcast to all gpus
    broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


if __name__ == '__main__':
    init_dist(launcher='pytorch')
    func = torch.compile(batch_shuffle_ddp)
    func(torch.ones(1, 1, 1, 1))