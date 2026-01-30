import torch
import time
import gc
import sys
import os
import numpy as np
import torchvision.models as models
import argparse

DEVICE = "cuda"

def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

def timed(model, example_inputs, times=1, dynamo=False):
    torch.manual_seed(1337)
    if dynamo:
        import sklearn
    gc.collect()
    synchronize()
    t0 = time.time_ns()
    # Dont collect outputs to correctly measure timing
    result = model(*example_inputs)
    synchronize()
    t1 = time.time_ns()
    return (t1 - t0) / 1_000_000

def speedup_experiment(model, example_inputs, model2, example_inputs2):
    repeat = 100
    timings = np.zeros((repeat, 2), np.float64)
    for rep in range(repeat):
        timings[rep, 0] = timed(model, example_inputs)
    for rep in range(repeat):
        timings[rep, 1] = timed(model2, example_inputs2, dynamo=True)
    median = np.median(timings, axis=0)
    print(f"Eager Latency: {median[0]} ms")
    print(f"sklearn Eager latency: {median[1]} ms")
    print(f"speedup: {median[0]/median[1]} ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="specify device")
    args = parser.parse_args()
    DEVICE = args.device
    model = models.alexnet(pretrained=True).to(DEVICE).half()
    example_inputs = (torch.randn(16, 3, 224, 224).to(DEVICE).half(), )
    model2 = models.alexnet(pretrained=True).to(DEVICE).half()
    example_inputs2 = (torch.randn(16, 3, 224, 224).to(DEVICE).half(), )
    model.eval()
    model2.eval()
    speedup_experiment(model, example_inputs, model2, example_inputs2)