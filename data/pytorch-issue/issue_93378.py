self._wrapped_model = torch.compile(self._wrapped_model)

import torch._inductor.config
torch._inductor.config.triton.cudagraphs = False