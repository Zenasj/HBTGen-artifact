"""To run:

pip install torch==2.4.1 accelerate==0.34.0 transformers==4.44.2

OMP_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=6,7 \
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=2 \
    fsdp_generate.py
"""
import accelerate
import torch
import torch.distributed
import torch.distributed._composable.fsdp
import transformers
import transformers.models.gpt_neo.modeling_gpt_neo


def main() -> None:
    torch.cuda.set_device(torch.device(torch.distributed.get_rank()))

    # creating a checkpoint we can load with load_checkpoint_and_dispatch
    transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path := "EleutherAI/gpt-neo-125m").save_pretrained(checkpoint := "checkpoint")

    # following the steps from: https://github.com/pytorch/torchtitan/blob/d2a4904f58accc683c17c66a360026cb3c8109af/docs/fsdp.md
    with torch.device("meta"):
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
        model = transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM(config)

    for submodule in model.modules():
        if isinstance(submodule, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock):
            torch.distributed._composable.fsdp.fully_shard(submodule)
    torch.distributed._composable.fsdp.fully_shard(model)

    model.to_empty(device="cuda")

    # this boils down to load_state_dict
    accelerate.load_checkpoint_and_dispatch(model, checkpoint, dtype=torch.bfloat16)


if __name__ == "__main__":
    torch.distributed.init_process_group(world_size=2)
    try:
        main()
    finally:
        torch.distributed.destroy_process_group()