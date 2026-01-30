import torch

def test_different_dtypes(self):
        # Test with tensors of different data types
        tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensor2 = torch.tensor([1, 2, 3], dtype=torch.float32)
        self.assertFalse(torch.equal(tensor1, tensor2))