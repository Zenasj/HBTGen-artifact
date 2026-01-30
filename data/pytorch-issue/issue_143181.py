import torch.nn as nn

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model_path = "/checkpoints/Qwen2-0.5B-Instruct/"

    if rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        param_init_fn = None
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path))
        param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)

    model = FSDP(
        model,
        sync_module_states=True,
        param_init_fn=param_init_fn,
        device_id=rank,
    )

    model = model.to('cpu')
    model = model.to('cuda')

    print(f"Rank {rank} Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)

if len(module_states) > 0:
        for idx, state in enumerate(module_states):
            dist.broadcast(state)

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from collections import defaultdict


def ensure_weights_retied(param_init_fn, model: torch.nn.Module, device: torch.cuda.device):
    """Handles `transformers` models with tied weights."""

    _tied_names = getattr(model, "_tied_weights_keys", None)
    if not _tied_names:
        # if no tied names just passthrough
        return param_init_fn

    # get map of parameter instances to params.
    # - needed for replacement later
    _tied_params = {}
    for name in _tied_names:
        name = name.split(".")
        name, param_name = ".".join(name[:-1]), name[-1]
        mod = model.get_submodule(name)
        param = getattr(mod, param_name)

        _tied_params[id(param)] = None  # placeholder for the param first

    # build param_init_fn for the case with tied params
    def param_init_fn_tied_param(module: torch.nn.Module):
        # track which params to tie
        # - usually only 1, but for completeness consider > 1
        params_to_tie = defaultdict(list)
        for n, param in module.named_parameters(recurse=False):
            if id(param) in _tied_params:
                params_to_tie[id(param)].append(n)

        # call the param init fn, which potentially re-allocates the
        # parameters
        module = param_init_fn(module)

        # search the parameters again and tie them up again
        for id_key, _param_names in params_to_tie.items():
            for param_name in _param_names:
                param = _tied_params[id_key]
                if param is None:
                    # everything will be tied to the first time the
                    # param is observed
                    _tied_params[id_key] = getattr(module, param_name)
                else:
                    setattr(module, param_name, param)  # tie

        return module

    return param_init_fn_tied_param

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model_path = "Qwen/Qwen2-0.5B-Instruct"

    if rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        param_init_fn = None
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path))
        param_init_fn = ensure_weights_retied(
            lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False),
            model=model,
            device=torch.cuda.current_device(),
        )


    model = FSDP(
        model,
        sync_module_states=True,
        param_init_fn=param_init_fn,
        device_id=rank,
    )

    model = model.to('cpu')
    model = model.to('cuda')

    print(f"Rank {rank} Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()