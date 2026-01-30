import torch.nn as nn

def test_load_state_dict_large(self):
        # construct a module with 4 levels of module, 10 linear each, leads to 10k items in the dictionary
        import copy
        import time
        base_module = nn.Linear(1,1)
        model = base_module
        for level in range(4):
           model = nn.Sequential(*[copy.deepcopy(model) for _ in range(10)])
        state_dict = model.state_dict()
        self.assertEqual(len(state_dict), 20000)
        st = time.time()
        model.load_state_dict(state_dict, strict=True)
        strict_load_time = time.time() - st
        # it took 0.5 seconds to 
        self.assertLess(strict_load_time, 10)