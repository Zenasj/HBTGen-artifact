import torch
import torch.nn as nn

def test_cpu_offload_full_param_padded(self):
    model = nn.Linear(5, 5).cuda()
    fsdp_model = FSDP(model)
    with FSDP.summon_full_params(fsdp_model, offload_to_cpu=True):
        ...

print(f"[Rank {self.rank}] _full_param_padded: {p._full_param_padded.device} {p._full_param_padded.storage().size()}")

print(f"[Rank {self.rank}] _full_param_padded: {p._full_param_padded.device} {p._full_param_padded.storage().size()}")

def test_cpu_offload_full_param_padded(self):
    model = nn.Sequential(
        nn.Linear(100, 100, bias=False),
        nn.Sequential(
            nn.Linear(100, 100, bias=False),
            nn.Sequential(
                nn.Linear(100, 100, bias=False),
                nn.Sequential(
                    nn.Linear(100, 100, bias=False),
                )
            )
        ),
    ).cuda()
    torch.cuda.empty_cache()
    print(f"[Rank {self.rank}] torch.cuda.memory_allocated: {torch.cuda.memory_allocated(self.rank)}")
    import functools
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.Sequential},
    )
    fsdp_model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
    if self.rank == 0:
        print(fsdp_model)
    torch.cuda.empty_cache()
    print(f"[Rank {self.rank}] torch.cuda.memory_allocated: {torch.cuda.memory_allocated(self.rank)}")
    torch.cuda.empty_cache()
    with FSDP.summon_full_params(fsdp_model, offload_to_cpu=True):
        torch.cuda.empty_cache()
        print(f"[Rank {self.rank}] torch.cuda.memory_allocated: {torch.cuda.memory_allocated(self.rank)}")