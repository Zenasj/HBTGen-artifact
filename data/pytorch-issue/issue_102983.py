import torch
import torch.nn as nn

def test_weight_norm_pickle(self):
        m = torch.nn.utils.weight_norm(nn.Linear(5, 7))
        print(m.weight)
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)
        print(m.weight)