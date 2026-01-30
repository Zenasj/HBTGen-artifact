import torch
import time
torch.set_num_threads(4)

a = torch.zeros((4, 96, 96))
b = torch.zeros((1, 4, 96, 96))
while True:
    b[0] = a
    time.sleep(.001)