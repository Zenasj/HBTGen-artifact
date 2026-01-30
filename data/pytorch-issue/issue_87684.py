import torch
with torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
    a = torch.rand(24, 128, 128)
    b = torch.rand(24, 128, 128, dtype=torch.bfloat16)
    c = torch.cat([a, b], 0)

import torch
torch._C._jit_set_autocast_mode(False)

def printTracedModel(model, input):
    with torch.no_grad(), torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
        traced = torch.jit.trace(model, input)
        fwd_graph = traced.graph_for(*input)
        print(fwd_graph)

def test_cat_check_type_promotion():
    def cat_along_dim(d):
        def forward_cat(*inputs):
            return torch.cat(inputs, d)
        return forward_cat

    x = torch.rand(24, 128, 128)
    y = torch.rand(24, 128, 128, dtype=torch.bfloat16)
    printTracedModel(cat_along_dim(0), [x, y])

if __name__ == '__main__':
    test_cat_check_type_promotion()

import torch
with torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
    a = torch.rand(24, 128, 128)
    b = torch.rand(24, 128, 128, dtype=torch.bfloat16)
    c = torch.cat([a, b], 0)
torch.allclose(c, torch.cat[a,b.to(dtype=torch.float),0]) # returns True

import torch
torch._C._jit_set_autocast_mode(False)

def printTracedModel(model, input):
    with torch.no_grad(), torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
        traced = torch.jit.trace(model, input)
        fwd_graph = traced.graph_for(*input)
        print(fwd_graph)

def testCatCheckTypePromotion():
    def cat_along_dim(d):
        def forward_cat(*inputs):
            return torch.cat(inputs, d)
        return forward_cat

    x = torch.rand(24, 128, 128)
    y = torch.rand(24, 128, 128, dtype=torch.bfloat16)
    printTracedModel(cat_along_dim(0), [x, y])

if __name__ == '__main__':
    testCatCheckTypePromotion()