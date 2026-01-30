import torch
import math

def get_score_mod_read_bias(bias: FloatTensor) -> _score_mod_signature:
    """
    Maximize the amount that we read into the score_mod kernel:
    bfloat16 [heads=6~64, ctx_len=512, ctx_len=512]
    you'd still hope this would be competitive with cutlassF/cuDNN
    """
    def score_mod(
        score: FloatTensor,
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> FloatTensor:
        return score + bias[h, q_idx, kv_idx]
    return score_mod

def get_score_mod_jump_table_and_emb(
    emb_weight: FloatTensor,
    num_buckets=32,
    max_distance=128,
    ctx_len=512,
) -> _score_mod_signature:
    """
    Minimize the amount that we read into the score_mod kernel to just the emb_weight and a jump table:
    bfloat16 [num_buckets=32, heads=6~64]
        int8 [ctx_len=512] # this would fit in uint4 if torch supported it
    it *is* possible to minimize the jump table's size to ~91 instead of 512, by introducing more offsets and fallbacks to the jump arithmetic.
    I have an implementation of that, and it was slightly slower. probably not worth the code complexity, but it had a cool property whereby you don't need to know ctx_len.
    """
    half_buckets = num_buckets // 2
    max_exact = half_buckets // 2
    relpos_to_bucket: LongTensor = torch.arange(ctx_len, device=emb_weight.device, dtype=torch.float32).div_(max_exact).log_().mul_((half_buckets - max_exact) / math.log(max_distance / max_exact)).long().clamp_max(max_exact-1).add_(max_exact).byte()
    relpos_to_bucket[:max_exact].copy_(torch.arange(max_exact, device=emb_weight.device))

    def score_mod(
        score: FloatTensor,
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> FloatTensor:
        relpos = (kv_idx - q_idx).abs_()
        relpos_buckets = relpos_to_bucket[relpos]
        relpos_buckets.add_(kv_idx > q_idx, alpha=half_buckets)
        return score + emb_weight[relpos_buckets.int(), h]
    return score_mod

def make_mask_mod(mask: BoolTensor) -> _mask_mod_signature:
    def mask_mod(
        batch: IntTensor,
        head: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        return mask[batch, kv_idx]
    return mask_mod

def make_mask_mod(mask: BoolTensor) -> _mask_mod_signature:
    def mask_mod(
        batch: IntTensor,
        head: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        return mask[batch, kv_idx] & mask[batch, q_idx]
    return mask_mod