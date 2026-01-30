import torch

from nemo.collections.asr.models import EncDecCTCModelBPE

model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")

model.to(device="cpu").freeze()
model = model.eval()

example_input = model.preprocessor.input_example(max_batch=2)

_ = torch._dynamo.export(model.preprocessor)(example_input)