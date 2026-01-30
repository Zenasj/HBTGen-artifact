import torch
import torch.nn as nn

py
with attention_context_parallel(), sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
    dquery = distribute_tensor(query, device_mesh, [Shard(2)])
    dkey = distribute_tensor(key, device_mesh, [Shard(2)])
    dvalue = distribute_tensor(value, device_mesh, [Shard(2)])
    
    dout: DTensor = torch.nn.functional.scaled_dot_product_attention(
        dquery, dkey, dvalue, is_causal=is_causal
    )
    out = dout.to_local()

py
with attention_context_parallel(), sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=nheads,
        dim_feedforward=dim,
        batch_first=True,
    ).to(dtype)
    encoder_layer = parallelize_module(
        module=encoder_layer,
        device_mesh=device_mesh,
        parallelize_plan={
            "self_attn": ContextParallel(),
        },
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)