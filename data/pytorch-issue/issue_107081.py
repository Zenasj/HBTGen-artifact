import torch

for n,p in FSDP_model.model.named_parameters():
            if p is not None:
                total_param = p.data.numel() 
                non_zero_param = torch.count_nonzero(p.data)
                env.print(n,'main train() percentage of zero weight:', (total_param-non_zero_param)/total_param, "total_param",total_param)

with FSDP.summon_full_params(
                FSDP_model,
                writeback=True,
                offload_to_cpu=True,
                rank0_only=False,
                with_grads=False
            ):
            for n,p in FSDP_model.model.named_parameters():
                       p.data.masked_fill_(mask, 0.0)
            for n,p in FSDP_model.model.named_parameters():
                      if p is not None:
                          total_param = p.data.numel() 
                          non_zero_param = torch.count_nonzero(p.data)
                          env.print(n,'within summon_full_params main train() percentage of zero weight:', (total_param-non_zero_param)/total_param, "total_param",total_param)