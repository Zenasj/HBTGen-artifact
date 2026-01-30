import torch

with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(fsdp_model):
    generated_text = fsdp_model.module.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_length=10,
    )