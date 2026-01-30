from torch.utils.pytree import tree_map

def somefunc():
    ret = tree_map(fn, tree)

import torch

def somefunc():
    ret = torch.utils.tree_map(fn, tree)