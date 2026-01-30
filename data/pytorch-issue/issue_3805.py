import torch.nn as nn

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True)
parser.add_argument("--dest", type=str, required=True)

args = parser.parse_args()

model_state = torch.load(args.source)
new_model_state = {}

for key in model_state.keys():
    new_model_state[key[7:]] = model_state[key]

torch.save(new_model_state, args.dest)

model = Model()
dp = nn.DataParallel(model)

# train dp
# ...

torch.save(model.state_dict(), path)
# or you can do torch.save(dp.module.state_dict(), path)

model = Model()
model.load_state_dict(torch.load(path))
dp = nn.DataParallel(model)