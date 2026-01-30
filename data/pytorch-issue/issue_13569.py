import torch.nn as nn

import torch
from torch import nn


def main():
    device = "cuda"
    embedding = nn.Embedding(128, 16, max_norm=1).to(device)
    batch = torch.randint(128, (32, ), device=device)

    embedding.forward(batch)
    print("OK")

    parallel = nn.DataParallel(embedding)
    parallel.forward(batch)


if __name__ == "__main__":
    main()