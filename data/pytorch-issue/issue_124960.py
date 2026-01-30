import torch

E = torch.einsum('pa, pb, abs, ps ->', xt, xt, EFS_coeff, spec_mask)

E = (spec_mask * (xt.unsqueeze(-1) * (xt.unsqueeze(-1).unsqueeze(-1) * EFS_coeff).sum(1)).sum(1)).sum()

def _add_file(filename: str) -> None:
    print('\nFILE TO TOKENIZE:')
    print(filename)
    try:
        with open(filename) as f:
            tokens = list(tokenize.generate_tokens(f.readline))
    except OSError:
        cache[filename] = {}
        return

def _tokenize(readline, encoding):
    lnum = parenlev = continued = 0
    numchars = '0123456789'
    contstr, needcont = '', 0
    contline = None
    indents = [0]

    if encoding is not None:
        if encoding == "utf-8-sig":
            # BOM will already have been stripped.
            encoding = "utf-8"
        yield TokenInfo(ENCODING, encoding, (0, 0), (0, 0), '')
    last_line = b''
    line = b''
    while True:                                # loop over lines in stream
        try:
            # We capture the value of the line variable here because
            # readline uses the empty string '' to signal end of input,
            # hence `line` itself will always be overwritten at the end
            # of this loop.
            last_line = line
            print(last_line)  # PRINT STATEMENT ADDED HERE
            line = readline()
        except StopIteration:
            line = b''

#!/usr/bin/env python
# coding: utf-8
"""
A functionally equivalent parser of the numpy.einsum input parser
"""

import itertools
from collections import OrderedDict

import numpy as np
...

#!/usr/bin/env python
# coding: utf-8

get_symbol(20000)
    #> '京'