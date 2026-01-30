import torch
import torch.nn.functional as F

def test_embedding_bag_input_error(self):
        input = torch.tensor([[-1, 0]], dtype=torch.int64).cuda()
        weight = torch.rand([2, 3], dtype=torch.float32).cuda()
        with self.assertRaises(RuntimeError):
            F.embedding_bag(input, weight)