import torch.nn as nn

import argparse
import threading
import datetime
import os
import random

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--size', default=64, type=int)
    parser.add_argument('--layers', default=4, type=int)
    parser.add_argument('--log-interval', default=100, type=int)
    parser.add_argument('--chkpt-interval', default=100, type=int)
    parser.add_argument('--total-iterations', default=1000000, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

    return parser.parse_args()


def abort():
    torch.distributed.distributed_c10d._abort_process_group(
        torch.distributed.distributed_c10d.GroupMember.WORLD
    )


def train(
    loop_iteration, base_store, model, opt, backend, device, timeout, args
):
    aborted = False

    log_interval = args.log_interval
    chkpt_interval = args.chkpt_interval

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Create a new Store by adding a prefix based on the current restart
    # iteration. PrefixStore wraps the baseline TCPStore which is reused for
    # all restart iterations
    store = torch.distributed.PrefixStore(str(loop_iteration), base_store)
    torch.distributed.distributed_c10d._store_based_barrier(
        rank,
        store,
        'initial',
        world_size,
        timeout=datetime.timedelta(seconds=60),
    )

    torch.distributed.init_process_group(
        backend,
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    local_rank = int(os.environ['LOCAL_RANK'])
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    random.seed((args.seed + loop_iteration) * world_size)
    fault_iteration = random.randint(1, 10)
    random.seed((args.seed + loop_iteration) * world_size + rank)
    delay = random.random() / 100

    print(f'{rank=} {fault_iteration=} {delay=}')

    for iteration in range(args.total_iterations):
        # Randomly trigger an example fault
        if iteration == fault_iteration and not aborted:
            aborted = True
            print(f'example fault at {iteration=} from {rank=}')

            # abort torch.distributed after a random delay
            timer = threading.Timer(
                delay,
                abort,
            )
            timer.start()

        inp = torch.rand(args.size, args.size).to(device)
        model.zero_grad()
        out = model_ddp(inp)
        loss = out.square().mean()
        loss.backward()
        opt.step()
        loss.item()

        if rank == 0 and iteration % log_interval == log_interval - 1:
            print(f'{rank=} {iteration=} {loss.item()=}')


def main():
    args = parse_args()
    print(f'{args}')

    local_rank = int(os.environ['LOCAL_RANK'])

    if args.device == 'cuda':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
        backend = 'nccl'
        timeout = datetime.timedelta(seconds=150)
    elif args.device == 'cpu':
        device = torch.device('cpu')
        backend = 'gloo'
        timeout = datetime.timedelta(seconds=10)
    else:
        raise RuntimeError

    # All objects created in ``main()`` are constructed only once, and reused
    # for all restart iterations.
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        *[torch.nn.Linear(args.size, args.size) for _ in range(args.layers)]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    # TCPStore uses ``(MASTER_PORT + 1)`` to avoid conflicts with TCPStore
    # created by ``torch.distributed.run`` and listening on ``MASTER_PORT``,
    store = torch.distributed.TCPStore(
        host_name=os.environ['MASTER_ADDR'],
        port=int(os.environ['MASTER_PORT']) + 1,
        world_size=int(os.environ['WORLD_SIZE']),
        is_master=(int(os.environ['RANK']) == 0),
        multi_tenant=True,
        wait_for_workers=True,
        use_libuv=True,
    )

    rank = int(os.environ['RANK'])
    loop_iteration = 0
    while True:
        print(f'Starting {loop_iteration=}')
        try:
            train(loop_iteration, store, model, opt, backend, device, timeout, args)
        except Exception as ex:
            print(f'Exception on {rank=} {str(ex)}')
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        loop_iteration += 1


if __name__ == '__main__':
    main()