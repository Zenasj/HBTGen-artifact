py
import cloudpickle
import torch

data = {"foo": lambda: True}
path = "foo.pickle"

with open(path, mode="wb") as file:
    cloudpickle.dump(data, file)  # works

torch.save(data, path, pickle_module=cloudpickle)  # fails