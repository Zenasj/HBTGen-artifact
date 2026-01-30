import torch

def test_repo(self):

        @torch.compile(dynamic=True, backend="inductor")
        def call(x, ref_id):
            self_id = 22
            if self_id == ref_id:
                x = torch.mul(x, 1.0)
            else:
                x = torch.mul(x, 0)
            return x

        x = torch.ones(2)
        self.assertEqual(call(x, 2), torch.zeros(2))