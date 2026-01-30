from dataclasses import is_dataclass

import torch


@torch.compile(fullgraph=True)
def func(x):
    if not is_dataclass(x):
        return x + 1

func(torch.ones(()))

from dataclasses import GenericAlias, _FIELDS
def is_dataclass_recode(obj):
    """Returns True if obj is a dataclass or an instance of a
    dataclass."""
    cls = obj if isinstance(obj, type) and not isinstance(obj, GenericAlias) else type(obj)
    return hasattr(cls, _FIELDS)

import torch


@torch.compile(fullgraph=True)
def func(x):
    if not is_dataclass_recode(x):
        return x + 1

func(torch.ones(()))