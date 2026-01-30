import torch
import torch.nn as nn

precision = MixedPrecision(
    param_dtype=torch.float32,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

# gather model states to rank 0 and offload to cpu
with FSDP.state_dict_type(
    model, 
    StateDictType.FULL_STATE_DICT, 
    FullStateDictConfig(
        offload_to_cpu=True, 
        rank0_only=True
    )
):
    gathered_model = model.state_dict()

    # gather optimizer states to rank 0
    gathered_optimizer = FSDP.optim_state_dict(
        model, 
        optimizer
    )

torch.save({
    "model": gathered_model,
    "optimizer": gathered_optimizer,
    "scaler": scaler.state_dict(),
    "lr_scheduler": lr_scheduler.state_dict(),
    "step": step,
}, checkpoint_path)

# gather model states to rank 0 and offload to cpu
with FSDP.state_dict_type(
    model, 
    StateDictType.FULL_STATE_DICT, 
    FullStateDictConfig(
        offload_to_cpu=True, 
        rank0_only=True
    )
):
    gathered_model = model.state_dict()

torch.save({
    "model": gathered_model,
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "lr_scheduler": lr_scheduler.state_dict(),
    "step": step,
}, checkpoint_path)

torch.distributed.barrier()
print(f"Checkpoint saved at {save_path}/model_checkpoint_step_{step}.pt")

parallelize_plan[f"model.layers.{i}.self_attn.q_proj"] = ColwiseParallel(
  _prepare_input=make_input_shard_1d
)
parallelize_plan[f"model.layers.{i}.self_attn.k_proj"] = ColwiseParallel(
  _prepare_input=make_input_shard_1d
)
parallelize_plan[f"model.layers.{i}.self_attn.v_proj"] = ColwiseParallel(
  _prepare_input=make_input_shard_1d
)
parallelize_plan[f"model.layers.{i}.self_attn.o_proj"] = RowwiseParallel(
  _prepare_input=make_input_shard_1d, 
  _prepare_output=make_output_reshard_tensor
)

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard


model = AutoModelForCausalLM.from_pretrained(
  "Research/Llama-2-7b-hf",
)

device_mesh = init_device_mesh(
    "cuda", 
    (1, 8), 
    mesh_dim_names=("dp", "tp")
)

print(device_mesh)

tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]

for model_block in model.model.layers:
    print(model_block)
    parallel = parallelize_module(
        module=model_block,
        device_mesh=tp_mesh,
        parallelize_plan={
            "self_attn.q_proj": ColwiseParallel(input_layouts=Shard(0)),
            "self_attn.k_proj": ColwiseParallel(input_layouts=Shard(0)),
            "self_attn.v_proj": ColwiseParallel(input_layouts=Shard(0)),
            "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(0)),
            "mlp.up_proj": ColwiseParallel(input_layouts=Shard(0)),
            "mlp.down_proj": RowwiseParallel(output_layouts=Shard(0)),
            "mlp.gate_proj": ColwiseParallel(input_layouts=Shard(0)),
        },
    )

# set mixed precision
mp_fsdp = self.mixed_precision_recipe()

# set auto wrap policy
def fsdp_auto_wrap_policy(
    self,
    transformer_block
):
    model_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            transformer_block,
        },
    )
    return model_auto_wrap_policy

model_auto_wrap_policy = self.fsdp_auto_wrap_policy(
    transformer_block=LlamaDecoderLayer,
)

# setup fsdp
model = FSDP(
    model, 
    auto_wrap_policy=model_auto_wrap_policy,
    mixed_precision=mp_fsdp,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device(),
    device_mesh=dp_mesh,
    forward_prefetch=True,
    use_orig_params=True,
)

# setup fsdp
model = FSDP(
    model, 
    auto_wrap_policy=model_auto_wrap_policy,
    mixed_precision=mp_fsdp,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=cpu_offload,
    device_id=torch.cuda.current_device(),
    device_mesh=dp_mesh if use_tp_sp else None,
    forward_prefetch=True,
    limit_all_gathers=True,
    use_orig_params=True,
)

# optimizer
optimizer = torch.optim.AdamW(
    decoupled_model_params,
    lr=learning_rate,
    betas=(0.90, 0.95),
    foreach=True,
)
        
def load_sharded_checkpoint(
    self,
    load_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
):
    step = 0
    # load model from a specific checkpoint name
    if load_path is not None:

        torch.distributed.barrier()

        with FSDP.state_dict_type(
            model, 
            StateDictType.SHARDED_STATE_DICT
        ):

            dist_dict = {
                "model": model.state_dict(),
                "optimizer": FSDP.optim_state_dict(model, optimizer),
                "lr_scheduler": lr_scheduler.state_dict(),
                "step": step,
            }

            load_state_dict(
                state_dict=dist_dict,
                storage_reader=FileSystemReader(load_path),
                planner=DefaultLoadPlanner(),
            )

            step = dist_dict["step"]

        torch.distributed.barrier()
    
    resume_step = step
    if get_rank() == 0 and step > 0:
        print(f"Resuming training from checkpoint at step {resume_step}.")
    return resume_step

def load_sharded_checkpoint(
    self,
    load_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
):
    step = 0
    # load model from a specific checkpoint name
    if load_path is not None:

        torch.distributed.barrier()

        with FSDP.state_dict_type(
            model, 
            StateDictType.SHARDED_STATE_DICT
        ):

            dist_dict = {
                "model": model.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "step": step,
            }

            load_state_dict(
                state_dict=dist_dict,
                storage_reader=FileSystemReader(load_path),
            )

            model.load_state_dict(dist_dict["model"])

            optimizer_state = load_sharded_optimizer_state_dict(
                model_state_dict=dist_dict["model"],
                optimizer_key="optimizer",
                storage_reader=FileSystemReader(load_path),
            )

            flattened_osd = FSDP.optim_state_dict_to_load(
                model, 
                optimizer, 
                optimizer_state["optimizer"]
            )

            optimizer.load_state_dict(flattened_osd)

            step = dist_dict["step"]

        torch.distributed.barrier()
    
    resume_step = step
    if get_rank() == 0 and step > 0:
        print(f"Resuming training from checkpoint at step {resume_step}.")
    return resume_step 
    
# setup fsdp
model = FSDP(
    model, 
    auto_wrap_policy=model_auto_wrap_policy,
    mixed_precision=mp_fsdp,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=cpu_offload,
    device_id=torch.cuda.current_device(),
    #device_mesh=dp_mesh if use_tp_sp else None,
    forward_prefetch=True,
    limit_all_gathers=True,
    use_orig_params=True,
)     

# load checkpoint if resuming
resume_step = self.load_sharded_checkpoint(
    load_path,
    model,
    optimizer,
    scheduler,
)