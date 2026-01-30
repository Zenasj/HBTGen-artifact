import torch
import numpy as np
import random

if args.seed:
    def worker_init_fn(x):
        seed = args.seed + x
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return
else: worker_init_fn = None

if args.seed: worker_init_fn=lambda x: [np.random.seed((args.seed + x)), random.seed(args.seed + x), torch.manual_seed(args.seed + x)]
else: worker_init_fn = None