import torch

@onlyCPU
def test_inplace_comparison_ops_require_inputs_have_same_dtype(self, device):
    with self.assertRaisesRegex(RuntimeError, 'Expected object of scalar type'):
        for op in ['lt_', 'le_', 'gt_', 'ge_', 'eq_', 'ne_', 'logical_xor_', 'logical_and_', 'logical_or_']:
            x = torch.tensor([1], dtype=torch.int)
            y = torch.tensor([2], dtype=torch.long)
            in_place_method = getattr(x, op)
            in_place_method(y)