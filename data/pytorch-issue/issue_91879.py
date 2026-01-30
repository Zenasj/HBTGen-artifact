import torch

model, optimizer = amp.initialize(model, optimizer,
                                      num_losses=len(task2scaler),
                                      enabled=opts.optimizer["fp16"], opt_level='O2')

model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank(), find_unused_parameters=False)
with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                    loss_id=task2scaler[name]) as scaled_loss:
    
    scaled_loss.backward()

model = FSDP(model,
                 auto_wrap_policy=t5_auto_wrap_policy,
                 mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    ),
                 device_id=torch.cuda.current_device(),
                 sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # ZERO2
                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE)