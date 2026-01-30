import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def test_muliple_pruning_calls_preserve_hooks():
    # Initial setup: a module with 3 pre-forward hooks where the middle one is
    # the hook for pruning.
    m = nn.Conv3d(2, 2, 2)
    m.register_forward_pre_hook(lambda x, y: y)
    prune.l1_unstructured(m, name='weight', amount=0.1)
    m.register_forward_pre_hook(lambda x, y: y)

    original_hooks = list(m._forward_pre_hooks.keys())
    prune.l1_unstructured(m, name='weight', amount=0.1)
    new_hooks = list(m._forward_pre_hooks.keys())

    print(original_hooks)
    print(new_hooks)

test_muliple_pruning_calls_preserve_hooks()