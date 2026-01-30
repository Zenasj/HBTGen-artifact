import torch
import transformers  # type: ignore[import]

real_model = transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2")
'h.0.attn.bias' in real_model.state_dict().keys()  # False (unexpected)

named_buffers = { k: v for k, v in real_model.named_buffers() }
'h.0.attn.bias' in named_buffers  # True

print(named_buffers['h.0.attn.bias'])  # torch.Tensor(...)