import torch.nn as nn

import torch, time
import torch.distributed as dist

def get_model_data(channels_last, rank=0):

    model = torch.nn.SyncBatchNorm(num_features=32)
    data = torch.rand((1,32,192,192,192))

    model=model.to(device=torch.device(rank))
    data=data.to(device=torch.device(rank))

    if channels_last:
        model=model.to(memory_format=torch.channels_last_3d)
        data=data.to(memory_format=torch.channels_last_3d)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    return model, data


def run_forward_backward(model, data, iter=30):
    with torch.cuda.amp.autocast(enabled=True):
        for _ in range(int(iter)):
            for param in model.parameters(): param.grad = None 
            logits = model(data)
            loss = torch.sum(logits**2)
            loss.backward()
            torch.distributed.barrier()
    
    return loss

def main_worker(rank, ngpus_per_node):

    print(f"rank {rank}")

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=ngpus_per_node, rank=rank)
    torch.backends.cudnn.benchmark = True
    n_iter =  30


    model, data = get_model_data(channels_last=False, rank=rank)
    run_forward_backward(model, data, iter=1) #warmup

    tic = time.time()
    run_forward_backward(model, data, iter=n_iter)
    if rank==0: print('DDP channels_last=False, run_forward_backward, time:', time.time()-tic, 'sec')


    ###############################
    model, data = get_model_data(channels_last=True, rank=rank)
    run_forward_backward(model, data, iter=1) #warmup

    tic = time.time()
    run_forward_backward(model, data, iter=n_iter)
    if rank==0: print('DDP channels_last=True, run_forward_backward, time:', time.time()-tic, 'sec')

    torch.distributed.destroy_process_group()


def main():
    ngpus_per_node = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

if __name__ == "__main__":
    main()