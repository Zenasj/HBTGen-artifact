import torch
import torch.nn as nn
import numpy as np

def test_params_are_unflattenned(self):
        layer_shape = (10, 12)
        model = nn.Linear(*layer_shape, bias=False).cuda(self.rank)
        fsdp_model = FSDP(deepcopy(model)).cuda(self.rank)

        flattened_param = fsdp_model.get_parameter("_fsdp_wrapped_module.flat_param")
        self.assertEqual(layer_shape[0] * layer_shape[1] / 2, flattened_param.numel())


        input = torch.randn(1, 10, device=self.rank)

        out = fsdp_model(input).sum()
        out.backward()
        with fsdp_model.summon_full_params():
            named_parameters = fsdp_model.named_parameters()
            if self.rank == 0:
                np = dict(named_parameters)
                for name, param in np.items():
                    print(f"name: {name} param grad {param.grad}")

                params = list(fsdp_model.parameters())
                for p in params:
                    print(f"param grad: {p.grad}")

            self.assertEqual(fsdp_model.weight.shape, model.weight.shape)