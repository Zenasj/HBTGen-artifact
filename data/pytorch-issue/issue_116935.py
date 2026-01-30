py
import torch
from x_transformers import ContinuousTransformerWrapper, Encoder


model = ContinuousTransformerWrapper(
    dim_in = 32,
    dim_out = 100,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 12,
        heads = 8,
        attn_flash = True,  # set to false to use LucidRains' attn impl instead of PyTorch's
       # can also use "with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):"
       # it's important to disable mem_efficient as well for some reason
    )
).cuda()

x = torch.randn((1, 1024, 32)).cuda()
z = torch.randn((1, 1024, 32)).cuda()

y = model(x) 

model = torch.compile(model, fullgraph=True)

y_2 = model(x)
z_2 = model(z)

import torch
from x_transformers import ContinuousTransformerWrapper, Encoder
import random

random.seed(0)
torch.manual_seed(0)


model = ContinuousTransformerWrapper(
    dim_in = 32,
    dim_out = 100,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 12,
        heads = 8,
        attn_flash = True,  # set to false to use LucidRains' attn impl instead of PyTorch's
       # can also use "with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):"
       # it's important to disable mem_efficient as well for some reason
    )
).cuda()

x = torch.randn((1, 1024, 32)).cuda()
z = torch.randn((1, 1024, 32)).cuda()

y = model(x)

model_compiled = torch.compile(model, fullgraph=True)

y_test = model_compiled(x)
print(torch.allclose(y, y_test, atol=1e-5))
print(torch.max(torch.abs(y - y_test)))

py

import torch
from x_transformers import ContinuousTransformerWrapper, Encoder
import random

random.seed(0)
torch.manual_seed(0)




with torch.autocast(device_type="cuda", dtype=torch.float16):

    model = ContinuousTransformerWrapper(
    dim_in = 32,
    dim_out = 100,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 12,
        heads = 8,
        attn_flash = True,  # set to false to use LucidRains' attn impl instead of PyTorch's
       # can also use "with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):"
       # it's important to disable mem_efficient as well for some reason
    )
    ).cuda()

    x = torch.randn((1, 1024, 32)).cuda()
    z = torch.randn((1, 1024, 32)).cuda()

    y = model(x)

    model_compiled = torch.compile(model, fullgraph=True)

    y_test = model_compiled(x)
    print(torch.allclose(y, y_test, atol=1e-5)) # Prints "False"
    print(torch.max(torch.abs(y - y_test))) # Prints "tensor(0.0020, device='cuda:0', dtype=torch.float16, grad_fn=<MaxBackward1>)"

py
inductor_decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,
        aten.arange,
        aten.bitwise_and_,
        aten.bitwise_or_,
        aten.clamp_min_,
        aten.dist,
        aten.empty_like,
        aten.flip,
        aten.gelu,
        aten.hardtanh,
        aten.index_select,
        aten.lcm,
        aten.leaky_relu,
        aten.linalg_vector_norm,
        aten._log_softmax,
        aten.max_pool2d_with_indices_backward,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten.native_batch_norm,
        aten.native_group_norm,
        # aten.native_layer_norm,
        aten._softmax,
        aten.sin_,
        aten.sqrt_,
        out_dtype,
        aten._to_copy,
        aten.tril_indices,
        aten.triu_indices,
        aten.upsample_bilinear2d.vec,
    ]
)

import torch
from x_transformers import ContinuousTransformerWrapper, Encoder
import random

random.seed(0)
torch.manual_seed(0)




lib = torch.library.Library("aten", "FRAGMENT")
autocast_cuda_ops = torch._C._dispatch_get_registrations_for_dispatch_key("AutocastCUDA")
ctr = 0
for op_name in autocast_cuda_ops:
    def autocast_nop(*args, **kwargs):
        with torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCUDA)):
            aten_op = kwargs.pop("aten_op")
            print("OP: " + str(aten_op))
            try:
                return aten_op(*args, **kwargs)
            except:
                breakpoint()
                return aten_op(*args, **kwargs)
    name = op_name[6:] # remove "aten::"
    base_name = name.split('.')[0]
    aten_op = getattr(torch.ops.aten, base_name)
    import functools
    autocast_impl = functools.partial(autocast_nop, aten_op=aten_op)

    if name or 'attention' in name:
        lib.impl(name, autocast_impl, "AutocastCUDA")


import contextlib
with torch.autocast(device_type="cuda", dtype=torch.float16):
#with contextlib.nullcontext():

    model = ContinuousTransformerWrapper(
    dim_in = 32,
    dim_out = 100,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 12,
        heads = 8,
        attn_flash = True,  # set to false to use LucidRains' attn impl instead of PyTorch's
       # can also use "with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):"
       # it's important to disable mem_efficient as well for some reason
    )
    ).cuda()

    x = torch.randn((1, 1024, 32)).cuda()
    z = torch.randn((1, 1024, 32)).cuda()

    y = model(x)

    model_compiled = torch.compile(model, fullgraph=True, backend="inductor")

    y_test = model_compiled(x)
    print(torch.allclose(y, y_test, atol=1e-5)) # Prints "False"
    print(torch.max(torch.abs(y - y_test))) # Prints "tensor(0.0020, device='cuda:0', dtype=torch.float16, grad_fn=<MaxBackward1>)"

import torch
from x_transformers import ContinuousTransformerWrapper, Encoder
import random

random.seed(0)
torch.manual_seed(0)

model = ContinuousTransformerWrapper(
    dim_in = 32,
    dim_out = 100,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 12,
        heads = 8,
        attn_flash = True,  # set to false to use LucidRains' attn impl instead of PyTorch's
       # can also use "with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):"
       # it's important to disable mem_efficient as well for some reason
    )
).cuda()

x = torch.randn((1, 1024, 32)).cuda()
z = torch.randn((1, 1024, 32)).cuda()

model_compiled = torch.compile(model, fullgraph=True)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    y = model(x)
    y_test = model_compiled(x)
y_ref = model(x.to(dtype=torch.float32)) # fp64 failed on the model for me, using fp32

print(f'amp eager vs ref difference: {str(torch.max(torch.abs(y_ref - y)))}')
print(f'amp compile vs ref difference: {str(torch.max(torch.abs(y_ref - y_test)))}')