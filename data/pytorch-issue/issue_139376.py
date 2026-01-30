import torch

def f(x):
            y = torch.sum(torch.sum(x, dim=-1))

            z = x / 10.0
            z_t = z.t().contiguous().t()
            return y, z, z_t

@triton.jit
def triton_unk_fused_add_sum_0(in_ptr0, in_ptr1, out_ptr0, ws_ptr, semaphores_ptr, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr, RSPLIT : tl.constexpr):
    xnumel = 1
    rnumel = 1048576
    rsplit_id = tl.program_id(0)
    num_rblocks = (rnumel + RBLOCK - 1) // RBLOCK
    rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
    rsplit_start = rsplit_chunk * rsplit_id
    rsplit_end = rsplit_chunk * (rsplit_id + 1)
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(rsplit_start, rsplit_end, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    if RSPLIT > 1:
        tmp4_ws = (ws_ptr + 0).to(tl.pointer_type(tl.float32))
        tl.store(tmp4_ws + (xindex * RSPLIT + rsplit_id), tmp4, None)
    if RSPLIT > 1:
        triton_helpers.gpu_barrier(semaphores_ptr + (2 * tl.program_id(1) + 0), RSPLIT, True)
    if RSPLIT > 1:
        tmp4_peers = tl.load(tmp4_ws + (xindex * RSPLIT + tl.arange(0, RSPLIT)[None,:]), None, eviction_policy='evict_first')
        tmp4 = tl.sum(tmp4_peers, 1)[:, None]
    if rsplit_id == (0 % RSPLIT):
        tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)