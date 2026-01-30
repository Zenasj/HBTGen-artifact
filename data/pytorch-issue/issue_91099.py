#!/usr/bin/env python

import torch
from torch._dynamo import optimize, config
config.cache_size_limit = 4

@optimize("inductor")
def toy_example(adict):
    # adict["text"] not used here
    return adict["data"].norm()

def test():
    adict = {"data": torch.ones(10), "text": ""}
    for i in range(5):
        adict["text"] = f"text-{i}"
        toy_example(adict)

if __name__ == "__main__":
    test()