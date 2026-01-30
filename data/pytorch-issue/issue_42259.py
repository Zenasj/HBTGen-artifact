import torch.nn as nn

import torch
from typing import Any, Dict

class IdListFeature(object):
    def __init__(self, lengths, values):
        self.lengths = lengths
        self.values = values


class IdScore(object):
    def __init__(self, ids, scores):
        self.ids = ids
        self.scores = scores


class IdScoreListFeature(object):
    def __init__(self, lengths, ids, scores):
        self.lengths = lengths
        self.values = IdScore(ids=ids, scores=scores)


class HashFeatureIds(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, module_input: Any) -> Any:
        if isinstance(module_input, IdListFeature):
            return module_input
        elif isinstance(module_input, IdScoreListFeature):
            return module_input
        raise Exception

m = HashFeatureIds()
torch.jit.script(m)