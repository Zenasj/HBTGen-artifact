import torch

def run(comm_group, args):
    """ Distributed function to be implemented later. """
    tensor = torch.ones(args.message_size).cuda()
    for i in range(200):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=comm_group)
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    tensor = torch.ones(args.message_size).cuda()
    tensor = tensor * args.rank
    torch.distributed.barrier(comm_group)
    start.record()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=comm_group)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end)
    print('Rank '+str(args.rank)+' has data '+str(tensor[0])+' time cost '+str(t)+' bandwidth '+str(args.message_size*4/t/1000/1000), flush=True)