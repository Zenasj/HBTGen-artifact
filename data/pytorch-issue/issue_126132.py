import torch

torch._inductor.config.post_grad_fusion_options = {
            "make_mmt_contiguous_pass": {},
            "pad_mm_pass": {},
            "decompose_mm_pass": {},
        }