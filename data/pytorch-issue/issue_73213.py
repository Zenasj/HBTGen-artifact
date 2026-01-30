import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn

mp = mp.get_context("forkserver") # Also tried with "spawn"

import torch.distributed as dist

import socket
from contextlib import closing

from torch import multiprocessing as mp


def find_free_port(address: str = "127.0.0.1") -> int:
    """Helper function to find a free port for distributed tests below"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    return port


def init_gru_and_run_on_single_input(worker_ind: int, port: int):
    print(f"Worker {worker_ind}: started")

    print(f"Worker {worker_ind}: creating TCP store")
    store = torch.distributed.TCPStore(
        host_name="127.0.0.1", port=port, world_size=2, is_master=worker_ind == 0,
    )

    print(f"Worker {worker_ind}: Starting process group")
    dist.init_process_group(
        backend="nccl", store=store, rank=worker_ind, world_size=2,
    )

    device = torch.device(worker_ind)

    print(f"Worker {worker_ind}: creating model")
    model = nn.GRU(128, 128, 1, True).to(device) # Also tried with nn.LSTM

    print(f"Worker {worker_ind}: running input")
    input = torch.zeros(1, 1, 128, device=device, dtype=torch.float32)

    # For `worker_ind==1` (which should have `model` on GPU 1) will result
    # in this using memory on GPU 0.
    model(input, None)

    print(f"Worker {worker_ind} entering 10 second sleep")
    time.sleep(10)
    dist.barrier()


if __name__ == "__main__":
    processes = []
    port = find_free_port()
    for i in range(2):
        processes.append(
            mp.Process(
                target=init_gru_and_run_on_single_input,
                kwargs={"worker_ind": i, "port": port},
            )
        )

        processes[-1].start()
        time.sleep(1)

    for p in processes:
        p.join()