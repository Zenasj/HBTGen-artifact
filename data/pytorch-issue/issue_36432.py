#!/usr/bin/env python

import torch

def parse_float_from_str(s: str) -> float:
    return float(torch.split(s, "_", -1)[0])

if __name__ == "__main__":
    scripted_f = torch.jit.script(parse_float_from_str)
    print(scripted_f("0_1"))