import torch as pt

pt.tensor(True, device="mps") * 0  # segmentation fault
pt.tensor(True, device="mps") * 1  # segmentation fault
pt.tensor(True, device="mps") + 0  # segmentation fault
pt.tensor(True, device="mps") + 1  # segmentation fault
pt.tensor(False, device="mps") + 0  # segmentation fault
pt.tensor(True, device="mps") * 2  # tensor(255, device='mps:0'), same for every number > 2 tested
pt.tensor(True, device="cpu") * 2  # tensor(2)