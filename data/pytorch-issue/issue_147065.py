import torch
import torch.nn as nn

class gul_grs_user_model(torch.nn.Module):
    def forward(self, xxx):
        # ...
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths)
        # ... other fbgemm ops ...
        return output

dynamicLib_path = torch._export.aot_compile(
       self.model,
       args=tuple(list(self._inputs_dict.values())),
       dynamic_shapes={**self._dynamic_shapes},
       options={
           "aot_inductor.output_path": os.path.join(self.dynamicLib_output_folder, dynamicLib_name),
           "max_autotune": True,
       },
   )