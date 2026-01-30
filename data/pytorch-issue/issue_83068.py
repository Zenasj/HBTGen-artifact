import torch
import torch.nn as nn

def test_mp_between_fwd_bwd(self):
    model = nn.Linear(5, 5).cuda()
    mp = MixedPrecision(torch.float16)
    fsdp_model = FSDP(model, mixed_precision=mp)  # single root FSDP instance
    out = fsdp_model(torch.randn((2, 5)).cuda())
    print(f"Model structure:\n{fsdp_model}")
    with FSDP.summon_full_params(fsdp_model):
        for param_name, param in fsdp_model.named_parameters():
            print(f"{param_name} {param.dtype}")

def test_mp_between_fwd_bwd(self):
    model = nn.Sequential(
        nn.Linear(5, 5),
        nn.Linear(5, 5),
    ).cuda()
    mp = MixedPrecision(torch.float16)
    fsdp_model = FSDP(
        model,
        mixed_precision=mp,
        auto_wrap_policy=always_wrap_policy,  # show for non-root FSDP instances too
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    out = fsdp_model(torch.randn((2, 5)).cuda())
    print(f"Model structure:\n{fsdp_model}")
    with FSDP.summon_full_params(fsdp_model):
        for param_name, param in fsdp_model.named_parameters():
            print(f"{param_name} {param.dtype}")