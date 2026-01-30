import torch
import traceback

devices = ['cpu', 'cuda']
ctors = [ torch.tensor, torch.Tensor ]

for src_dev in devices:
    for dst_dev in devices:
        for src_ctor in ctors:
            for dst_ctor in ctors:
                try:
                    a = src_ctor(1, device = src_dev)
                    b = dst_ctor(a, device = dst_dev)
                    if a == b: print("OK")
                except:
                    traceback.print_exc()

img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))