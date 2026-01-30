import dataclasses

import torch


def is_dataclass_instance(obj):
    if isinstance(obj, type):
        return False
    cls = type(obj)
    return hasattr(cls, "__dataclass_fields__")


def apply_to_collection(data):
    if is_dataclass_instance(data):
        fields = {}
        for k, v in fields.items():
            try:
                pass
            except dataclasses.FrozenInstanceError as e:
                break
    return data


def foo():
    apply_to_collection(torch.tensor(1.))


training_step = torch.compile(foo)
training_step()

import torch

def make(g):
    @torch.compile(backend='eager')
    def f():
        while True:
            try:
                print(g)
            except Exception as _:
                break
    return f

make(None)()