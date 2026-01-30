import os
import torch
import torch.distributed as dist

def run(devices):
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    assert len(devices) == world_size
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    device = devices[rank]
    if device != torch.device('cpu'):
        torch.cuda.set_device(device)
    for _ in range(10):
        if rank == 0:
            # Send the tensor to process 1
            tensor0 = torch.rand(1000000, device=device)
            send_req = dist.isend(tensor0, dst=1)
            # Receive the tensor from process 0
            tensor1 = torch.zeros(1000000, device=device)
            recv_req = dist.irecv(tensor1, src=1)
        else:
            # Send the tensor to process 0
            tensor1 = torch.rand(1000000, device=device)
            send_req = dist.isend(tensor1, dst=0)
            # Receive the tensor from process 1
            tensor0 = torch.zeros(1000000, device=device)
            recv_req = dist.irecv(tensor0, src=0)
        send_req.wait()
        recv_req.wait()
        if rank==0:
            print(rank, tensor0.mean(), tensor1.mean())
        if rank==1:
            print(rank, tensor0.mean(), tensor1.mean())


def init_processes(fn, devices):
    """ Initialize the distributed environment. """
    dist.init_process_group('mpi')
    fn(devices)


if __name__ == "__main__":
    devices = [torch.device('cuda:0'), torch.device('cuda:1')]
    init_processes(run, devices)