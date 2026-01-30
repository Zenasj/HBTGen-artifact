import torch.nn as nn

import tempfile

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcpsd


def reload_same_model():
    """Save and reload a model and its optimizer."""
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters())
    for n, p in model.named_parameters():
        print(f"{n:<15}: {p.norm(p=1).item()}")
    to_save = {
        "iteration": 10,
        "model": dcpsd.get_model_state_dict(model),
        "optimizer": dcpsd.get_optimizer_state_dict(model, optimizer),
    }
    tmp_dir = tempfile.mkdtemp()
    dcp.save(to_save, storage_writer=dcp.filesystem.FileSystemWriter(tmp_dir))

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters())
    to_load = {
        "iteration": -1,
        "model": dcpsd.get_model_state_dict(model),
        "optimizer": dcpsd.get_optimizer_state_dict(model, optimizer),
    }
    dcp.load(to_load, storage_reader=dcp.filesystem.FileSystemReader(tmp_dir))
    dcpsd.set_model_state_dict(model, to_load["model"])
    dcpsd.set_optimizer_state_dict(model, optimizer, to_load["optimizer"])
    for n, p in model.named_parameters():
        print(f"{n:<15}: {p.norm(p=1).item()}")
    print(to_load["iteration"])


def load_into_smaller_model():
    """Save a model and load it into a smaller model."""
    model = Model()
    for n, p in model.named_parameters():
        print(f"{n:<15}: {p.norm(p=1).item()}")
    to_save = {
        "iteration": 10,
        "model": dcpsd.get_model_state_dict(model),
    }
    tmp_dir = tempfile.mkdtemp()
    dcp.save(to_save, storage_writer=dcp.filesystem.FileSystemWriter(tmp_dir))

    model = SmallerModel()
    to_load = {
        "iteration": -1,
        "model": dcpsd.get_model_state_dict(model),
    }
    dcp.load(to_load, storage_reader=dcp.filesystem.FileSystemReader(tmp_dir))
    dcpsd.set_model_state_dict(model, to_load["model"])  # No need for StateDictOptions(strict=False), why?
    for n, p in model.named_parameters():
        print(f"{n:<15}: {p.norm(p=1).item()}")
    print(to_load["iteration"])


def load_into_bigger_model():
    """Save a model and load it into a bigger model."""
    model = Model()
    for n, p in model.named_parameters():
        print(f"{n:<15}: {p.norm(p=1).item()}")
    to_save = {
        "iteration": 10,
        "model": dcpsd.get_model_state_dict(model),
    }
    tmp_dir = tempfile.mkdtemp()
    dcp.save(to_save, storage_writer=dcp.filesystem.FileSystemWriter(tmp_dir))

    model = BiggerModel()
    to_load = {
        "iteration": -1,
        "model": dcpsd.get_model_state_dict(model),
    }
    dcp.load(to_load, storage_reader=dcp.filesystem.FileSystemReader(tmp_dir))
    dcpsd.set_model_state_dict(model, to_load["model"])
    for n, p in model.named_parameters():
        print(f"{n:<15}: {p.norm(p=1).item()}")
    print(to_load["iteration"])


class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(2, 4), torch.nn.Linear(4, 8))


class SmallerModel(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(2, 4))


class BiggerModel(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(2, 4), torch.nn.Linear(4, 8), torch.nn.Linear(8, 16))


if __name__ == "__main__":
    reload_same_model()  # OK
    load_into_smaller_model()  # OK, doesn't warn about "unexpected" keys
    load_into_bigger_model()  # RuntimeError: Missing key in checkpoint state_dict: model.2.weight