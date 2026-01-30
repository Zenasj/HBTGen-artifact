import torch

from torch.multiprocessing import Process, Event, Value
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch import device as torch_device, save, bfloat16
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from os import environ
from functools import partial
from gc import collect

model_id = 'codellama/CodeLlama-7b-Instruct-hf'
output_dir = 'out'
file_name = 'sharded'
world_size = 4
port = 9999
seed = 29

def load_and_save(rank, model, auto_wrap_policy, event, counter):

    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = str(port)
    init_process_group("nccl", rank=rank, world_size=world_size)

    print(f'wrapping in rank {rank}')

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch_device(f'cuda:{rank}'),
    )

    with counter.get_lock():
        counter.value += 1
        if counter.value == world_size:
            event.set()

    barrier()

    if rank == 0:
        print('done wrapping fsdp_model')

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
        print(f'creating cpu state in rank {rank}')
        cpu_state = fsdp_model.state_dict()
        print(f'created cpu state in rank {rank}')

    if rank == 0:
        print("now saving")
        save(cpu_state, f'{output_dir}/{file_name}.pt')
        print("saved")

    destroy_process_group()

if __name__ == '__main__':
    
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=bfloat16,
    )

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    processes = []

    event = Event()
    counter = Value('i', 0)

    try:
        for rank in range(world_size):
            p = Process(
                target=load_and_save,
                args=(rank, model, auto_wrap_policy, event, counter),
            )
            p.start()
            processes.append(p)

        event.wait()
        print('deleting model in main process')
        del model
        collect()
    
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        for p in processes:
            p.kill()
            print(f'killed process {p.pid}')

set_start_method('spawn', force=True)