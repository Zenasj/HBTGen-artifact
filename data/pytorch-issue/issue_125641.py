# based on the PyTorch DDP example

import argparse
import logging
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

torch.set_float32_matmul_precision("high")


def pytorch_logs_to_file(file: str = "pytorch.log"):
    torch._logging.set_logs(
        dynamo=logging.INFO,
        aot=logging.INFO,
        inductor=logging.INFO,
        dynamic=logging.INFO,
        distributed=logging.INFO,
        graph_breaks=True,
        guards=True,
        recompiles=True,
        recompiles_verbose=True,
        output_code=True,
        graph_code=True,
        graph=True,
        ddp_graphs=True,
    )
    torch._logging._init_logs(file)

    loggers = logging.Logger.manager.loggerDict.keys()
    for logger_name in loggers:
        if logger_name.startswith("torch"):
            logger = logging.getLogger(logger_name)
            if isinstance(logger, logging.Logger):
                handlers = logger.handlers
                for handler in handlers:
                    if isinstance(handler, logging.StreamHandler):
                        logger.removeHandler(handler)


class EmbedHeadModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: Tensor):
        out = self.vocab_embed(x)
        out = self.head(out)
        return out


def get_batch(
    batch_size: Tensor, sequence_length: int, vocab_size: int, device: torch.device, dynamic: bool
) -> Tuple[Tensor, Tensor]:
    if dynamic:
        input = torch.randint(
            0,
            vocab_size - 1,
            (batch_size, sequence_length - random.randint(0, min(512, sequence_length / 2)) // 8 + 1),
            device=device,
        )
    else:
        input = torch.randint(0, vocab_size - 1, (batch_size, sequence_length + 1), device=device)
    return input[:, :-1].contiguous(), input[:, 1:].contiguous()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--variable_seqlen", action="store_true", help="Batches have variable sequence lengths")
    parser.add_argument("--dynamic_true", action="store_true", help="Compile with dynamic=True instead of None")
    parser.add_argument(
        "--use_mark_dynamic", action="store_true", help="Use torch._dynamo.mark_dynamic for dynamic shapes"
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--log_name", type=str, default="pytorch.log")

    return parser.parse_args()


def train():
    args = parse_args()

    if args.ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
    else:
        rank = 0

    if args.logging and rank == 0:
        pytorch_logs_to_file(args.log_name)

    device_id = rank % torch.cuda.device_count()
    model = EmbedHeadModel(args.vocab_size, args.hidden_size).to(device=device_id)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.compile:
        model = torch.compile(model, dynamic=True if args.dynamic_true and not args.use_mark_dynamic else None)

    if args.ddp:
        model = DistributedDataParallel(model, device_ids=[device_id])

    model.train()

    for _ in range(0, args.iterations):
        data, targets = get_batch(
            args.batch_size, args.sequence_length, args.vocab_size, device_id, args.variable_seqlen
        )
        if args.use_mark_dynamic:
            torch._dynamo.mark_dynamic(data, index=1)

        output = model(data)
        loss = criterion(output.view(-1, args.vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()

class EmbedHeadModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: Tensor):
        out = self.vocab_embed(x)
        out = self.head(out)
        return out