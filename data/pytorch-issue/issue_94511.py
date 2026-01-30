import os
import argparse
import functools
import torch
from itertools import chain
import torch.nn as nn
import torch.optim as optim
from transformers import (
    OPTForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTAttention
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    MixedPrecision,
    FullyShardedDataParallel as FSDP
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)


def getDataset():
    raw_datasets = load_dataset("wikitext", "wikitext-2-v1")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= 1024:
            total_length = (total_length // 1024) * 1024
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + 1024]
                for i in range(0, total_length, 1024)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {1024}",
    )

    return lm_datasets["train"]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(args, model, rank, world_size, train_loader, optimizer, epoch):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(rank)
        attention_mask = batch["attention_mask"].to(rank)
        labels = batch["labels"].to(rank)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
        optimizer.zero_grad()
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(input_ids)
        if rank == 0:
            print(batch_idx, " *"*10)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch, ddp_loss[0] / ddp_loss[1]))


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    train_dataset = getDataset()
    train_loader = DataLoader(
        train_dataset, collate_fn=default_data_collator,
        batch_size=1, num_workers=1
    )

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100000
    )
    # my_auto_wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy, transformer_layer_cls={
    #         OPTDecoderLayer, OPTAttention, nn.LayerNorm, nn.Linear}
    # )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    if rank == 0:
        print("*"*10+"loading to cpu"+"*"*10)
    model = OPTForCausalLM.from_pretrained("facebook/opt-30b")
    model = checkpoint_wrapper(model, offload_to_cpu=True)

    model = FSDP(model,
                 cpu_offload=CPUOffload(CPUOffload(offload_params=True)),
                 auto_wrap_policy=my_auto_wrap_policy,
                 mixed_precision=MixedPrecision(param_dtype=torch.float16,
                                                reduce_dtype=torch.float16,
                                                buffer_dtype=torch.float16,
                                                keep_low_precision_grads=True)
                 )
    if rank == 0:
        print("*"*10+"print the fsdp model"+"*"*10)
        print(model)
        print_file = open("./model", 'w')
        print(model, file=print_file)
        print()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader,
              optimizer, epoch)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch OPT Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)

model = OPTForCausalLM.from_pretrained(model_name)
for i, layer in enumerate(model.model.decoder.layers):
   model.model.decoder.layers[i] = checkpoint_wrapper(layer, offload_to_cpu=True)