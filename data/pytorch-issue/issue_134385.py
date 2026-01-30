import torch

torch.utils.flop_counter.FlopCounterMode

def run_flex_decoding_causal(device: str = "cuda"):
    dtype_flops_utilization_map = {
        torch.bfloat16: "0.8",
    }
    B = 4
    H = 16
    D = 64
    q_len = [32]
    kv_len = [1024, 8192]
    results = []

    def causal_mask(batch, head, token_q, token_kv):
        return token_q >= token_kv
    
    for dtype, expected_flops_utilization in dtype_flops_utilization_map.items():
        flops_utilization = 0
        for seqlen_q in q_len:
            for seqlen_kv in kv_len:
                query = torch.randn((B, H, seqlen_q, D), dtype=dtype, device=device, requires_grad=True)
                key = torch.randn((B, H, seqlen_q, D), dtype=dtype, device=device, requires_grad=True)
                value = torch.randn((B, H, seqlen_q, D), dtype=dtype, device=device, requires_grad=True)
                block_mask = create_block_mask(causal_mask, 1, 1, seqlen_q, seqlen_kv)

                with FlopCounterMode(display=False) as mode:
                    flex_attention(query, key, value, block_mask=block_mask)
                
                flops = mode.get_total_flops()

                compiled_fn = torch.compile(flex_attention, dynamic=False)

                for _ in range(WARMUP_ITER):
                    compiled_fn(query, key, value, block_mask=block_mask)

                us_per_iter = benchmarker.benchmark_gpu(lambda: compiled_fn(query, key, value, block_mask=block_mask)) * 1000
                flops_utilization += us_per_iter * flops / 1e9 / A100_40G_BF16_TFLOPS
        
        flops_utilization = flops_utilization / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "flex_decoding_causal",
                "flops_utilization",
                expected_flops_utilization,
                f"{flops_utilization:.02f}",
                dtype_str,
                device,
            )
        )
    
    return results