import torch.nn as nn

if __name__ == "__main__":
    t1 = nn.ConstantPad1d((3, 1), 3.5)
    t2 = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
    t3 = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)