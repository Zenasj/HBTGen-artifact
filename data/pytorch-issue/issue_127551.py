import torch
import torch.nn as nn

def test_tbe_compile_default(self) -> None:
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        device = "cuda"
        m = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.MANAGED if device == "cuda" else EmbeddingLocation.HOST,
                    ComputeDevice.CUDA if device == "cuda" else ComputeDevice.CPU,
                )
                for (E, D) in zip(Es, Ds)
            ],
        )
        m.train(True)
        x = torch.Tensor([[[1], [1]], [[3], [4]]]).to(dtype=torch.int64, device=device)
        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=device == "cpu")
        out = m(indices, offsets, total_unique_indices=indices.numel())
        loss = reduce_to_scalar_loss(out)
        loss.backward()
        torch.compile(m, backend="inductor", fullgraph=True)(
            indices, offsets, total_unique_indices=indices.numel()
        )

class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[0, 0]", primals_5: "i32[0]", primals_6: "f32[0]", primals_7: "f32[16000]", primals_8: "i32[2]", primals_9: "i64[2]", primals_10: "i32[3]", primals_11: "i64[3]", getitem_1: "i64[4]", getitem_2: "i64[5]", tangents_1: "f32[2, 16]"):
        # File: /data/users/ivankobzarev/fbsource/buck-out/v2/gen/fbcode/acb493b09f6f34db/torchrec/distributed/tests/__test_pt2__/test_pt2#link-tree/fbgemm_gpu/split_embedding_codegen_lookup_invokers/lookup_sgd.py:66 in invoke, code: return torch.ops.fbgemm.split_embedding_codegen_lookup_sgd_function(
        auto_functionalized_1 = torch._higher_order_ops.auto_functionalize.auto_functionalized(torch.ops.fbgemm.split_embedding_backward_codegen_sgd_unweighted_exact_cuda.default, grad_output = tangents_1, dev_weights = primals_6, uvm_weights = primals_7, lxu_cache_weights = primals_4, weights_placements = primals_8, weights_offsets = primals_9, D_offsets = primals_10, max_D = 8, hash_size_cumsum = primals_11, total_hash_size_bits = 11, indices = getitem_1, offsets = getitem_2, pooling_mode = 0, lxu_cache_locations = primals_5, unused_ = 32, max_segment_length_per_warp = 32, stochastic_rounding = True, info_B_num_bits = 26, info_B_mask_int64 = 67108863, use_uniq_cache_locations = False, use_homogeneous_placements = True, learning_rate = 0.01);  tangents_1 = primals_4 = primals_8 = primals_9 = primals_10 = primals_11 = getitem_1 = getitem_2 = primals_5 = None
        getitem_6: "f32[0]" = auto_functionalized_1[1]
        getitem_7: "f32[16000]" = auto_functionalized_1[2];  auto_functionalized_1 = None
        
        # No stacktrace found for following nodes
        copy_: "f32[0]" = torch.ops.aten.copy_.default(primals_6, getitem_6);  primals_6 = getitem_6 = None
        copy__1: "f32[16000]" = torch.ops.aten.copy_.default(primals_7, getitem_7);  primals_7 = getitem_7 = None
        return [None, None, None, None, None, None, None, None, None, None, None, None, None]