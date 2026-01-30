import torch.nn as nn

py
"""
Standalone test file for max_autotune with captured buffers.

This file brings together the test along with all supporting functions,
decorators, and helper classes so that you can run it as a standalone file.
"""

import functools
from collections import namedtuple
from typing import Callable, Optional
import unittest
from unittest import skipUnless
from unittest.mock import patch

import torch
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16

# Imports from inductor and attention modules
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
    noop_mask,
)
from torch.utils._triton import has_triton

# -------------------- Global definitions --------------------


Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")

# Aliases
index = torch.ops.aten.index
Tensor = torch.Tensor

# Create attention partial function
def create_attention(score_mod, block_mask, enable_gqa=False):
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )

def create_block_mask_test(score_mod, query, key):
    block_mask = create_block_mask(
        score_mod, 1, 1, query.shape[-2], key.shape[-2], query.device
    )
    return block_mask

# Test dtypes and page sizes
test_dtypes = (
    [torch.float16, torch.bfloat16, torch.float32]
    if PLATFORM_SUPPORTS_BF16
    else [torch.float16, torch.float32]
)
test_dtypes_fast = [torch.float16]
test_page_sizes = [64, 128, 256]

# --------- Useful score mod functions for testing ---------

# Dimensions for test tensors
B = 4
S = 2048
D = 64
(Hq, Hkv) = (16, 8)

test_Hq_Hkv = [
    (16, 1),
    (8, 2),
    (16, 16),
]

test_Bq_Bkv = [
    (3, 1),
    (5, 1),
    (8, 1),
    (16, 1),
]

test_block_size = [
    64,
    128,
    (1, 64),
    (128, 64),
]

# Helper function to clone query, key, value tensors
def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype = None,
):
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    query_ref = query.detach().clone().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.detach().clone().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.detach().clone().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref

# Helper function to reserve batch entries in paged attention
def batch_reserve(paged_attention: PagedAttention, target_seq_len: torch.Tensor):
    (B,) = target_seq_len.shape
    for b in range(B):
        paged_attention.reserve(
            torch.tensor(b),
            target_seq_len[b],
        )

# -------------------- Test Class Definition --------------------

class TestFlexDecoding(InductorTestCase):
    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: Optional[str] = None,
    ):
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() and not torch.isnan(ref_error).any():
            self.assertTrue(False, "Output/Grad with NaN")
        if ref_error < (1e-4) * golden_out.abs().mean():
            print(
                "very small ref error of ",
                (ref_error.to(torch.float64) * (1e5) / golden_out.abs().mean()),
            )
            tolerance = Tolerances(atol=2e-1, rtol=2e-1)
            torch.testing.assert_close(
                golden_out.to(dtype=compiled_out.dtype),
                compiled_out,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
        elif compiled_error > ref_error * fudge_factor:
            name = tensor_name if tensor_name is not None else ""
            msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
            self.assertTrue(False, msg)

    def _check_out(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
    ):
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note: when using float32 the softmax computation might be less accurate
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")


    def preprocess_paged_attention(
        self,
        score_mod: Optional[Callable],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_mask,
        dtype: torch.dtype = torch.float16,
        page_size: int = 128,
    ):
        assert block_mask is not None, "Must provide block_mask"
        Q_B, Q_H, Q_S, _ = q.shape
        KV_B, KV_H, KV_S, QK_D = k.shape
        _, _, _, V_D = v.shape

        # Use a larger batch size for testing
        max_batch_size = max(Q_B, KV_B) + 3
        n_pages = (KV_S + page_size - 1) // page_size * max_batch_size

        MAX_CACHED_SEQ_LEN = n_pages * page_size
        k_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            QK_D,
            device="cuda",
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            V_D,
            device="cuda",
            dtype=dtype,
        )

        # "Randomly" initialize the page table
        paged_attention = PagedAttention(n_pages, page_size, max_batch_size)
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 4, KV_S // 3], device="cuda"),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 2, KV_S // 2], device="cuda"),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 2, KV_S, KV_S // 2, KV_S], device="cuda"),
        )
        batch_reserve(
            paged_attention, torch.tensor([KV_S, KV_S, KV_S, KV_S], device="cuda")
        )

        input_pos = (
            torch.arange(KV_S, device="cuda", dtype=torch.int32)
            .unsqueeze(0)
            .expand(KV_B, KV_S)
        )
        batch_idx = torch.arange(KV_B, device="cuda", dtype=torch.int32)
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        converted_block_mask = paged_attention.convert_logical_block_mask(block_mask)
        converted_score_mod = paged_attention.get_score_mod(score_mod)

        return k_cache, v_cache, converted_block_mask, converted_score_mod

    def run_paged_attention(
        self,
        score_mod: Optional[Callable],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dtype: torch.dtype = torch.float16,
        block_mask: Optional[BlockMask] = None,
    ):
        Q_B, Q_H, _ = q.shape[0], q.shape[1], k.shape[1]
        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, 1, S)

        (
            k_cache,
            v_cache,
            converted_block_mask,
            converted_score_mod,
        ) = self.preprocess_paged_attention(
            score_mod, q, k, v, block_mask, dtype, block_mask.BLOCK_SIZE[1]
        )

        compiled_sdpa = torch.compile(flex_attention)

        compiled_out, compiled_lse = compiled_sdpa(
            q,
            k_cache,
            v_cache,
            return_lse=True,
            block_mask=converted_block_mask,
            score_mod=converted_score_mod,
            enable_gqa=(not q.shape[1] == k.shape[1]),
        )
        return compiled_out, compiled_lse

    def run_test_with_paged_attention(
        self,
        score_mod: Optional[Callable],
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = Hq,
        Q_S: int = 1,
        QK_D: int = D,
        KV_B: int = B,
        KV_H: int = Hkv,
        KV_S: int = S,
        V_D: int = D,
        block_mask: Optional[BlockMask] = None,
    ):
        assert Q_H % KV_H == 0

        q = torch.randn(
            (Q_B, Q_H, Q_S, QK_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, QK_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, 1, KV_S)

        sdpa_partial = create_attention(
            score_mod, block_mask, enable_gqa=(not Q_H == KV_H)
        )
        golden_out, gold_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)

        compiled_out, compiled_lse = self.run_paged_attention(
            score_mod, q, k, v, dtype, block_mask
        )

        self._check_out(golden_out, ref_out, compiled_out)
        self._check_out(gold_lse, ref_lse, compiled_lse)

    # -------------------- The Test Method --------------------
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune_with_captured(self):
        # Create captured buffers
        head_scale = torch.randn(Hq, device="cuda")
        batch_scale = torch.randn(B, device="cuda")
        tok_scale = torch.randn(S, device="cuda")
        q_scale = torch.randn(1, device="cuda")

        def bias_mod(score, batch, head, token_q, token_kv):
            score = score + tok_scale[token_kv]
            score = score + q_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test_with_paged_attention(bias_mod)

# -------------------- Main --------------------

if __name__ == "__main__":
    # Run the test using unittest's CLI.
    unittest.main()