import os
import torch
import torch.nn as nn

if __name__ == '__main__':
    model = nn.Linear(3, 32)
    model.eval()

    os.makedirs("ドキュメント", exist_ok=True)
    with open("ドキュメント/checkpoint.pth", mode="wb") as f:
        torch.save(model.state_dict(), f)