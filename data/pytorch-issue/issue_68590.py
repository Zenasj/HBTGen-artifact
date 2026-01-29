# Input is a tuple of two sparse COO tensors of shape (2, 2), dtype=torch.float32
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        actual, expected = inputs

        # Old strategy: coalesce both and compare indices and values
        a_old = actual.coalesce()
        e_old = expected.coalesce()
        old_ok = a_old.indices().equal(e_old.indices()) and a_old.values().equal(e_old.values())

        # New strategy: check coalesced status first, then coalesce if needed, then check nnz, indices, values
        # Assuming check_is_coalesced is True (default)
        if actual.is_coalesced() != expected.is_coalesced():
            new_ok = False
        else:
            # If not coalesced, coalesce both
            if not actual.is_coalesced() or not expected.is_coalesced():
                a_new = actual.coalesce()
                e_new = expected.coalesce()
            else:
                a_new, e_new = actual, expected
            # Check nnz, indices, values
            new_ok = (a_new._nnz() == e_new._nnz()) and a_new.indices().equal(e_new.indices()) and a_new.values().equal(e_new.values())

        # Return whether the two strategies differ
        return torch.tensor(old_ok != new_ok, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Create actual and expected tensors as in the example
    actual_indices = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.int64)
    actual_values = torch.tensor([1., 1., 1.], dtype=torch.float32)
    actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2), dtype=torch.float32)

    expected_indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    expected_values = torch.tensor([1., 2.], dtype=torch.float32)
    expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2), dtype=torch.float32)

    return (actual, expected)

