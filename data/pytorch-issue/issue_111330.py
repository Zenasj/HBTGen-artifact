import torchvision

# test code
import tracemalloc
import psutil
import gc
import torch
from torchvision.models import vit_l_32

f = open("tracemalloc_pytorch.txt", "w")
process = psutil.Process()
tracemalloc.start()
model = vit_l_32(weights=None)
torch.save(model.state_dict(), "model.pt")

for i in range(1000):
    tracemalloc.reset_peak()

    torch.load("model.pt")

    size, peak = tracemalloc.get_traced_memory()
    log = f"iter {i+1}: current size: {size}, peak size: {peak}, threshold: {gc.get_threshold()}, cnt: {gc.get_count()}, mem: {process.memory_info().rss}\n"
    print(log)
    f.write(log)

tracemalloc.stop()
f.close()

# test code
import tracemalloc
import psutil
import gc

import torch
from torch.serialization import _open_file_like
from torchvision.models import resnet18

f = open("tracemalloc_streamreader.txt", "w")
process = psutil.Process()
tracemalloc.start()

model = resnet18(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model.pt")

for i in range(100000):
    tracemalloc.reset_peak()

    with _open_file_like("model.pt", "rb") as opened_file:
        opened_zipfile = torch._C.PyTorchFileReader(opened_file)
        del opened_zipfile

    size, peak = tracemalloc.get_traced_memory()
    log = f"iter {i+1}: current size: {size}, peak size: {peak}, threshold: {gc.get_threshold()}, cnt: {gc.get_count()}, mem: {process.memory_info().rss}\n"
    print(log)
    f.write(log)

tracemalloc.stop()
f.close()

import tracemalloc
import psutil
import gc
import random
import torch
import pickle
from torch.serialization import _open_file_like, _open_zipfile_reader, _is_zipfile, _legacy_load
from torchvision.models import resnet18

f = open("tracemalloc_streamreader.txt", "w")
process = psutil.Process()
tracemalloc.start()

model = resnet18(weights="IMAGENET1K_V1")
# torch.save(model.state_dict(), "model.pt")
torch.save(model.state_dict(), "model.pt", _use_new_zipfile_serialization=False)  # legacy

def test():
    tracemalloc.reset_peak()

    with _open_file_like("model.pt", "rb") as opened_file:
        if _is_zipfile(opened_file):
            type_ = "zipfile"
            with _open_zipfile_reader(opened_file) as opened_zipfile:  # leak!
                pass
        else:
            type_ = "legacy"
            _legacy_load(opened_file, map_location=torch.device("cpu"), pickle_module=pickle)  # leak!

    size, peak = tracemalloc.get_traced_memory()
    log = f"iter {i+1}: type: {type_}, current size: {size}, peak size: {peak}, threshold: {gc.get_threshold()}, cnt: {gc.get_count()}, mem: {process.memory_info().rss}\n"
    print(log)
    f.write(log)


for i in range(10000):
    test()

tracemalloc.stop()
f.close()

import tracemalloc
import psutil
import gc
import random
import torch
from torch.serialization import _open_file_like, _open_zipfile_reader
from torchvision.models import resnet18

f = open("tracemalloc_streamreader.txt", "w")
process = psutil.Process()
tracemalloc.start()

model = resnet18(weights="IMAGENET1K_V1")
# torch.save(model.state_dict(), "model.pt")
torch.save(model.state_dict(), "model.pt", _use_new_zipfile_serialization=False)  # legacy

def test():
    tracemalloc.reset_peak()

    with _open_file_like("model.pt", "rb") as opened_file:
        # with _open_zipfile_reader(opened_file) as opened_zipfile:  # leak!
        with _open_zipfile_reader("model.pt") as opened_zipfile:  # no leak!
            pass

    size, peak = tracemalloc.get_traced_memory()
    log = f"iter {i+1}: type: {type_}, current size: {size}, peak size: {peak}, threshold: {gc.get_threshold()}, cnt: {gc.get_count()}, mem: {process.memory_info().rss}\n"
    print(log)
    f.write(log)


for i in range(10000):
    test()

tracemalloc.stop()
f.close()

import torch


class _open_file(torch.serialization._opener):
    def __init__(self, name, mode):
        super().__init__(open(name, mode))
        self.name = name

    def __exit__(self, *args):
        self.file_like.close()


class _open_zipfile_reader(torch.serialization._opener):
    def __init__(self, name_or_buffer) -> None:
        name_or_buffer = name_or_buffer if not hasattr(name_or_buffer, "name") else name_or_buffer.name
        super().__init__(torch._C.PyTorchFileReader(name_or_buffer))


torch.serialization._open_file = _open_file
torch.serialization._open_zipfile_reader = _open_zipfile_reader