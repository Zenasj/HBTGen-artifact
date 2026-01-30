import torch.nn as nn

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch as ch
import numpy as np
import random

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    ch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed=seed)

    ch.cuda.manual_seed(seed)
    ch.cuda.manual_seed_all(seed)
    ch.backends.cudnn.deterministic = True

    ch.backends.cudnn.benchmark = False
    ch.backends.cuda.matmul.allow_tf32 = True # TODO: check
    ch.backends.cudnn.allow_tf32 = True # TODO: check
    ch.backends.cudnn.enabled = True

    ch.backends.cuda.enable_flash_sdp(False)
    ch.backends.cuda.enable_mem_efficient_sdp(False)
    ch.backends.cuda.enable_math_sdp(True)

set_seed(0)

def test_fn(primals, fn, x):
    def grads(*params):
        def g(params):
            y = fn(x, *params)
            return y.sum()

        return ch.func.grad(g)(params)

    tangents = tuple(ch.randn_like(p) for p in primals)

    output = grads(*primals)
    jvp_output, _ = ch.func.jvp(grads, primals, tangents)

    if not ch.is_tensor(output):
        output = ch.cat([o.flatten() for o in output], 0)
        jvp_output = ch.cat([o.flatten() for o in jvp_output], 0)

    print('norm(output - jvp_output)', (output - jvp_output).norm())


### TEST LAYERNORM
params = (ch.randn(20).cuda(), ch.randn(20).cuda())
x = ch.randn(100, 20).cuda()

def layer_norm(x, ln_weight, ln_bias):
    return ch.nn.functional.layer_norm(x, (20,), weight=ln_weight, bias=ln_bias)

print('LAYERNORM')
test_fn(params, layer_norm, x)

### TEST LINEAR
params = (ch.randn(20, 20).cuda(),)
def linear(x, w):
    return ch.nn.functional.linear(x, w)

print('LINEAR')
test_fn(params, linear, x)