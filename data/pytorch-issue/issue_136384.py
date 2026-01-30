import torch
import torch._dynamo.config as config

x = 5

def access_global():
    global x
    if x == 5:
        pass

def access_config():
    if torch._dynamo.config.compiled_autograd:
        pass

def access_config2():
    if config.compiled_autograd:
        pass

def access_state():
    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        pass

import sys
import timeit
print("local file global:", min(timeit.repeat(access_global)))
print("dynamo config global: ", min(timeit.repeat(access_config)))
print("dynamo config global 2:", min(timeit.repeat(access_config2)))
print("dynamo file global", min(timeit.repeat(access_state)))