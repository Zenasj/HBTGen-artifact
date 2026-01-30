import torch

if rank == 0:
    model = ...
    param_init_fn = None
else:
    with torch.device("meta"):
        model = ...
    param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)

model = FSDP(model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        mixed_precision=mixed_precision,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}),
        limit_all_gathers=True,
        device_id=dev,
        param_init_fn=param_init_fn
    )