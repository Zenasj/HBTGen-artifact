import torch.nn as nn

# _jit_internal.py

def is_final(ann):
    return ann.__module__ in {'typing', 'typing_extensions'} and \
        (getattr(ann, '__origin__', None) is Final)