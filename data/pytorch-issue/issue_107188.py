import torch
import torchaudio
from torchaudio.io import StreamReader

bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH

feature_extractor = bundle.get_streaming_feature_extractor()
decoder = bundle.get_decoder()
token_processor = bundle.get_token_processor()

# random values
# This works
decoder(torch.randn(80, 80), length=torch.tensor([1.0]), beam_width=10)

decoder = torch.compile(decoder, fullgraph=True)

# This does not work
decoder(torch.randn(80, 80), length=torch.tensor([1.0]), beam_width=10)

score = (torch.stack(torch.tensor(_get_hypo_score(h_b))).logaddexp(append_blank_score)).item()

import torch
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(backend='eager', fullgraph=True)
def f(v):
    return torch.tensor([v.item()])

f(torch.randn(1))