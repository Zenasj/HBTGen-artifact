import torch

# fx_graph_runnable.py
...
args = [((2304, 768), (768, 1), torch.float32, 'cuda'), ((2304,), (1,), torch.float32, 'cuda'), ((s0, s1, 768), (768*s1, 768, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
...

# fx_graph_runnable.py

# before
...
args = [((2304, 768), (768, 1), torch.float32, 'cuda'), ((2304,), (1,), torch.float32, 'cuda'), ((s0, s1, 768), (768*s1, 768, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
...

# after
...
args = []
args.append(rand_strided((2304, 768), (768, 1), torch.float32, 'cuda'))  # shape (2304, 768), stride (768, 1)
args.append(rand_strided((2304,), (1,), torch.float32, 'cuda'))  # shape (2304,), stride (1,)
args.append(rand_strided((160, 49, 768), (37632, 768, 1), torch.float32, 'cuda'))  # shape (s0, s1, 768), stride (768*s1, 768, 1)
...

# bw that inputs have SymInt
...
args = []
args.append(160)  # s0
args.append(49)  # s1
args.append(rand_strided((7840, 768), (768, 1), torch.float32, 'cuda'))  # shape (s0*s1, s3), stride (s3, 1)
args.append(rand_strided((160, 6, 49, 128), (37632, 6272, 128, 1), torch.float32, 'cuda'))  # shape (s0, s4, s1, s5), stride (s1*s4*s5, s1*s5, s5, 1)
...