import torch
import torch.distributed.elastic.multiprocessing


@torch.distributed.elastic.multiprocessing.errors.record
def Main():
    torch.distributed.init_process_group("nccl")
    the_main_process_group = torch.distributed.new_group(ranks=None, backend=None)
    torch.distributed.barrier(group=the_main_process_group)


if __name__ == "__main__":
    assert torch.distributed.is_available()
    assert torch.distributed.is_torchelastic_launched()

    Main()