import torch.nn as nn

import os
import sys
import math
import logging
import torch
import torch.distributed as dist
from typing import List
from itertools import chain
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import (
    default_data_collator,
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
)
logger = logging.getLogger(__name__)

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)

def build_datasets(tokenizer, phase):
    dataset_name = "wikitext"
    dataset_config_name="wikitext-103-raw-v1"
    cache_dir="./.cache/wikitext/wikitext103"
    validation_split_percentage = 5
    raw_datasets = load_dataset(
        dataset_name, dataset_config_name, cache_dir=cache_dir
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{validation_split_percentage}%:]",
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warning(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default"
            " `block_size` value of 1024. If you would like to use a longer `block_size` up to"
            " `tokenizer.model_max_length` you can override this with `--block_size xxx`."
        )
        block_size = 1024

    # Main data processing function that will concatenate all texts from our dataset and generate
    # chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it
        # instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together,
    # so group_texts throws away a remainde for each of those groups of 1,000 texts.
    # You can adjust that batch_siz here but a higher value might be slowe to preprocess.
    #
    # To speed up this part, we use multiprocessing.
    # See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=16,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets[phase]

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(f"local_rank={local_rank}, rank={rank}, world_size={world_size}")
    
    device = torch.device("cuda", local_rank)
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(local_rank)

    ### build model and tokenizer ###
    model_name_or_path = "gpt2"
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        low_cpu_mem_usage=False
    )
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()

    print_rank_0(model)
    model = model.to(device)

    # wrapper DDP
    if world_size > 1:
        model = DDP(model, device_ids=[device], output_device=device)

    # wrapper torch compile
    # torch._logging.set_logs(dynamo=logging.INFO)
    # torch._dynamo.config.verbose = True

    def comstom_backend(
        gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ):
        # gm.graph.print_tabular()
        return gm.forward

    torch._dynamo.reset()
    model = torch.compile(model, backend=comstom_backend)

    ### build data loader ###
    # Downloading and loading a dataset from the hub.
    train_dataset = build_datasets(tokenizer, "train")
    eval_dataset = build_datasets(tokenizer, "test")
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=False,
        batch_size=16,
        sampler=DistributedSampler(train_dataset),
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        shuffle=False,
        batch_size=16,
        sampler=DistributedSampler(eval_dataset),
    )

    ### start eval ###
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(eval_dataloader):
            # H2D
            batch_data["input_ids"] = batch_data["input_ids"].to(device)
            batch_data["attention_mask"] = batch_data["attention_mask"].to(device)
            batch_data["labels"] = batch_data["labels"].to(device)
            logger.info(f"[rank {rank}][{idx}/{len(eval_dataloader)}]: batch_data['input_ids'].shape={batch_data['input_ids'].shape}")
           
            # forward
            outputs = model(**batch_data)
            logger.info(f"[rank {rank}][{idx}/{len(eval_dataloader)}]: workers well")