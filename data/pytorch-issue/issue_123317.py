import torch


def fn(x):
    del x["a"]


inputs = {"a": torch.tensor([1]), "b": torch.tensor([1])}
compiled = torch._dynamo.export(fn)(inputs)