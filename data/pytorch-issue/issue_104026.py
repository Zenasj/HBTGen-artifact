import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import transformers
import os
import functools
import getpass


def get_cache_dir():
    if os.path.exists("/scr-ssd/"):
        return f"/scr-ssd/{getpass.getuser()}"
    else:
        return f"/scr/{getpass.getuser()}"


def init_distributed(rank, world_size, master_addr='localhost', master_port=12355, backend='nccl'):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run(local_rank, world_size):
    init_distributed(local_rank, world_size)

    model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={transformers.models.gpt_neox.GPTNeoXLayer})
    if local_rank == 0:
        model = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m', cache_dir=get_cache_dir())
        param_init_fn = None
    else:
        ## OPTION 1: (crashes on model(**batch).loss)
        # with torch.device('meta'):
        #     model = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m', cache_dir=get_cache_dir())

        ## OPTION 2: (hangs on model(**batch).loss)
        config = transformers.AutoConfig.from_pretrained('EleutherAI/pythia-160m', cache_dir=get_cache_dir())
        with torch.device('meta'):
            model = transformers.AutoModelForCausalLM.from_config(config)
        param_init_fn = lambda mod: mod.to_empty(device=f'cuda:{local_rank}', recurse=False)

    model = FSDP(model,
                auto_wrap_policy=model_auto_wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=local_rank,
                sync_module_states=True,
                param_init_fn=param_init_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    for idx in range(10):
        batch = {
            'input_ids': torch.arange(512).repeat(4, 1).to(local_rank) + 1000,
            'attention_mask': torch.ones(512).repeat(4, 1).to(local_rank),
            'labels': torch.arange(512).repeat(4, 1).to(local_rank) + 1000,
        }
        if local_rank == 0:
            print(batch['input_ids'].shape, end='\r')
        loss = model(**batch).loss
        if local_rank == 0:
            print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if local_rank == 0:
            print(f'{idx} done')


def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run, nprocs=world_size, args=(world_size,), join=True)

if __name__ == '__main__':
    main()