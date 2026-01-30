import torch.nn.functional as F

import torch
import torchaudio
import torchaudio.functional as F

bundle = torchaudio.pipelines.MMS_FA

SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
waveform, _ = torchaudio.load(SPEECH_FILE)
TRANSCRIPT = "i had that curiosity beside me at this moment".split()
LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)
tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

def align(emission, tokens, device):
    emission = emission.to(device)
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

for device in ["cpu", "cuda:0", "cuda:1"]:
    print(f'Running on: {device}')
    model = bundle.get_model(with_star=False).to(device)
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))

    
    aligned_tokens, alignment_scores = align(emission, tokenized_transcript, device=device)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])
    print(word_spans)
    print()