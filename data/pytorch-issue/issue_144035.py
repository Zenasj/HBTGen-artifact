import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset


DIM = 256  # Success at 128, failure at 256
SEQ_LEN = 32


@torch.compile(fullgraph=True)
def mlp_forward(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    b1: Tensor | None = None,
    b2: Tensor | None = None,
) -> Tensor:
    y = F.linear(x, w1, b1)
    y = F.relu(y)
    y = F.linear(y, w2, b2)
    return y


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ):
        super().__init__()
        self.checkpoint = True
        self.w_in = nn.Parameter(torch.randn(hidden_features, in_features))
        self.w_out = nn.Parameter(torch.randn(out_features, hidden_features))
        self.b_in = nn.Parameter(torch.randn(hidden_features))
        self.b_out = nn.Parameter(torch.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpoint:
            result = checkpoint(
                mlp_forward,
                x,
                self.w_in,
                self.w_out,
                self.b_in,
                self.b_out,
                use_reentrant=False,
            )
        else:
            result = mlp_forward(x, self.w_in, self.w_out, self.b_in, self.b_out)
        assert isinstance(result, Tensor)
        return result


def main(ddp=True):
    print(f"Running with DDP: {ddp}, DIM: {DIM}, SEQ_LEN: {SEQ_LEN}")
    x = torch.randn(100, SEQ_LEN, DIM)
    y = torch.zeros(100)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)
    model = MLP(DIM, 4 * DIM, DIM)

    if ddp:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="nccl", world_size=1, rank=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    device = torch.device("cuda:0")
    model = model.to(device)

    if ddp:
        model = nn.parallel.DistributedDataParallel(model)

    model.train()

    try:
        for batch in dataloader:
            x, y = batch
            x = x.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
    finally:
        if ddp:
            dist.destroy_process_group()

    print("Success")


if __name__ == "__main__":
    main(ddp=True)  # Fails

    # Running first without DDP followed by DDP makes the DDP version work.
    # Maybe triggering compiles outside DDP is key?
    # main(ddp=False)
    # main(ddp=True)