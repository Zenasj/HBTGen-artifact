import torch
import torch.nn as nn

import psutil, torch, transformers, gc, os, sys
import math

# Size in MB
model_size = 512

kB = 1024
MB = kB * kB
precision_size = 4 # bytes per float
activation_size = math.floor(math.sqrt(model_size * MB / precision_size))

class Net(torch.nn.Module):
    def __init__(self, activation_size):
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(activation_size, activation_size)
    def forward(self, x):
        return {"result": self.linear(x)}

def collect_and_report(s):
    gc.collect()
    print(s)
    #print("psutil: ", psutil.virtual_memory().percent)
    print("CPU MB used by this process: ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    print("GPU MB allocated by pytorch: ", torch.cuda.memory_allocated(0) / 1024 ** 2)
    print()

def run_test(device_str):
    device = torch.device(device_str)
    dummy_input = torch.zeros(activation_size, requires_grad=True).to(device)

    collect_and_report("Before loading model: ")
    model = Net(activation_size).to(device)
    collect_and_report("After loading model: ")

    torch.onnx.export(model, dummy_input, "dummy.onnx")
    collect_and_report("After exporting model: ")

    del model
    collect_and_report("After deleting model:")

print("Running CPU test: ")
run_test("cpu")

print("Running GPU test: ")
run_test("cuda")