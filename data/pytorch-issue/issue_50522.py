#!/usr/bin/env python3
import argparse
import time
import torch.distributed.rpc as rpc
import torchvision


def get_image(path='Path to large image'):
    print('Remote get image: ' + path)
    return torchvision.io.read_image(path)


def run(args):
    if args.rank == 0:
        for i in range(1000000000):
            time.sleep(10)
    else:
        for i in range(1000000000):
            target_worker_name = 'worker_{}'.format(0)
            frames = rpc.rpc_sync(target_worker_name, get_image, args=())
            print(frames.size())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument('-i',
                        '--init-method',
                        type=str,
                        default='env://',
                        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=2, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    args = parser.parse_args()
    print(args)

    worker_name = 'worker_{}'.format(args.rank)
    rpc.init_rpc(worker_name,
                 rank=args.rank,
                 world_size=args.world_size,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=8,
                                                                     rpc_timeout=300))

    run(args)


if __name__ == '__main__':
    main()

def get_image():
    return torch.rand(3, 224, 224)