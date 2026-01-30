import torch.nn as nn

import argparse
import torch
import sys

parser = argparse.ArgumentParser(prog='grid', description='grid')
parser.add_argument('--precision', action="store", type=str, required=False, default='int32')
args = parser.parse_args()

print('precision: ' + args.precision)
print('torch.__version__: ' + torch.__version__)
print('torch.xpu.is_available(): ' + str(torch.xpu.is_available()))
props = torch.xpu.get_device_properties()
print('xpu.has_fp16: ' + str(props.has_fp16))
print('xpu.has_fp64: ' + str(props.has_fp64))

if args.precision == 'bfloat16':
    input = torch.empty(1, 1, 2, 2, dtype=torch.bfloat16, device=torch.device('xpu'))
    grid = torch.empty(1, 1, 1, 2, dtype=torch.bfloat16, device=torch.device('xpu'))
elif args.precision == 'float16':
    input = torch.empty(1, 1, 2, 2, dtype=torch.float16, device=torch.device('xpu'))
    grid = torch.empty(1, 1, 1, 2, dtype=torch.float16, device=torch.device('xpu'))
elif args.precision == 'float32':
    input = torch.empty(1, 1, 2, 2, dtype=torch.float32, device=torch.device('xpu'))
    grid = torch.empty(1, 1, 1, 2, dtype=torch.float32, device=torch.device('xpu'))
elif args.precision == 'float64':
    input = torch.empty(1, 1, 2, 2, dtype=torch.float64, device=torch.device('xpu'))
    grid = torch.empty(1, 1, 1, 2, dtype=torch.float64, device=torch.device('xpu'))
else:
    print('unsupported precision: ' + args.precision)
    sys.exit(1)

res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
print(res)