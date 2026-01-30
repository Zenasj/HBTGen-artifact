import torch
import pathlib
torch.save(torch.rand(int((2 * 1024**3) / 4), dtype=torch.float32, device="cpu"),
           pathlib.Path("pytorch_test_load.temp"))
torch.load(pathlib.Path("pytorch_test_load.temp"))