import torch

def test_mutable_list_remove_tensor(self):
        def test_list_remove_tensor():
            a = [torch.ones(2), torch.zeros(2), torch.ones(2)]
            a.remove(torch.zeros(2))

            return len(a) == 2
        self.checkScript(test_list_remove_tensor, ())