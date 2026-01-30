import networkx as nx  # if this line is removed, it works !!
import torch
import pickle
import io

def _reconstruct_pickle(obj):
    f = io.BytesIO()
    pickle.dump(obj, f)
    f.seek(0)
    obj = pickle.load(f)
    f.close()
    return obj

def test():
    x = torch.float32
    new_bg = _reconstruct_pickle(x)

test()