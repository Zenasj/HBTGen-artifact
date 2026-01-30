import functools
import os
import sys

import torch
from tabulate import tabulate
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Block


@record
def main():
    sdtype = sys.argv[1]
    state_dict_type = {
        "full": StateDictType.FULL_STATE_DICT,
        "local": StateDictType.LOCAL_STATE_DICT,
        "sharded": StateDictType.SHARDED_STATE_DICT,
    }[sdtype]

    rank = int(os.environ["RANK"])

    sharding_strategy = ShardingStrategy.FULL_SHARD

    t5_model = T5ForConditionalGeneration(
        T5Config(
            d_ff=512,
            d_kv=32,
            d_model=128,
            is_encoder_decoder=True,
            model_type="t5",
            n_positions=512,
            num_heads=2,
            num_layers=4,
            vocab_size=32128,
        )
    )

    fsdp_model = FullyShardedDataParallel(
        t5_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={T5Block},
        ),
        sharding_strategy=sharding_strategy,
        device_id=device_id,
    )

    table = []
    layer_names = ["shared.weight", "encoder.embed_tokens.weight"]

    with FullyShardedDataParallel.state_dict_type(
        fsdp_model, state_dict_type=state_dict_type
    ):
        for layer_name in layer_names:
            state_dict = fsdp_model.state_dict()

            layer = state_dict.get(layer_name)
            tensor_type = type(layer)
            row = {
                "rank": rank,
                "sharding strategy": sharding_strategy.name,
                "state_dict_type": state_dict_type.name,
                "layer": layer_name,
            }

            if layer is None:
                continue

            row.update(
                {
                    "dtype": layer.dtype,
                    "shape": layer.shape,
                    "tensor type": tensor_type.__qualname__,
                }
            )

            if tensor_type != ShardedTensor:
                row["storage"] = layer.untyped_storage()

            table.append(row)

    if rank == 0:
        print(tabulate(table, headers="keys", stralign="left"))


if __name__ == "__main__":
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    dist.init_process_group("nccl")

    try:
        main()
    finally:
        dist.destroy_process_group()