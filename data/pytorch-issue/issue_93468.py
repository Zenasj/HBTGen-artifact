import torch
import torch.nn as nn

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

class encoder(nn.Module):

    def __init__(self):
        super(encoder, self).__init__()
        pass
    
    def forward(self, a):
    
        b = ~sequence_mask(a)
        
        return b
        
rawmodel = encoder().cuda()

model = torch.compile(rawmodel, fullgraph=True, backend='nvprims_aten')

a = torch.randint(10, 30, (10,)).cuda()

model(a)

import torch
import torch.nn as nn

def sequence_mask(lengths, max_len):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

class encoder(nn.Module):

    def __init__(self):
        super(encoder, self).__init__()
        pass
    
    def forward(self, a, max_a):

        b = ~sequence_mask(a, max_a)
        
        return b
        
rawmodel = encoder().cuda()

model = torch.compile(rawmodel, fullgraph=True, backend='nvprims_aten')

a = torch.randint(10, 30, (10,)).cuda()
max_a = a.max().item()
model(a, max_a)

import torch
import torch.nn as nn

import torch._dynamo.config
import torch._inductor.config
import logging
#torch._dynamo.config.log_level = logging.DEBUG
#torch._dynamo.config.output_code = True
torch._inductor.config.debug = True

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return ~(torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

class encoder(nn.Module):

    def __init__(self):
        super(encoder, self).__init__()
        pass
    
    def forward(self, a):
    
        b = sequence_mask(a)
        
        return b
        
rawmodel = encoder().cuda()

model = torch.compile(rawmodel, dynamic=False)

a = torch.randint(10, 30, (10,)).cuda()

model(a)

@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.int64) + -9223372036854775808
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & (_tmp1 < tmp0), tmp0, _tmp1)
    tmp1 = tl.max(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + 0 + tl.zeros([XBLOCK, 1], tl.int32), tmp1, None)

@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.int64) + -9223372036854775808
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & (_tmp1 < tmp0), tmp0, _tmp1)
    tmp1 = tl.max(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + 0 + tl.zeros([XBLOCK, 1], tl.int32), tmp1, None)

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*i1', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 23
    x1 = (xindex // 23)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp2 = tmp0 < tmp1
    tmp3 = tmp2 == 0
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp2 = tmp0 < tmp1
    tmp3 = tmp2 == 0
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)

torch._dynamo.config.specialize_int = False