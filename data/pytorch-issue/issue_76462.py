import torch

def test_normalize_quantized_eb(self):
        target = torch.ops.quantized.embedding_bag_byte_rowwise_offsets
        args = (
            torch.empty((2, 3), dtype=torch.uint8),
            torch.empty((2,), dtype=torch.int64),
            torch.empty((2,), dtype=torch.int64),
        )
        norm_args_and_kwargs = normalize_function(
            target, args, normalize_to_only_use_kwargs=True
        )
        self.assertTrue(norm_args_and_kwargs is not None)
        self.assertEqual(
            set(norm_args_and_kwargs.kwargs.keys()),
            {
                "weight",
                "indices",
                "offsets",
                "scale_grad_by_freq",
                "mode",
                "pruned_weights",
                "per_sample_weights",
                "compressed_indices_mapping",
                "include_last_offset",
            },
        )
        self.assertEqual(norm_args_and_kwargs.args, tuple())