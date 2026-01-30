from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model = model.cuda()

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
input_values = inputs["input_values"].cuda()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
with torch.no_grad():
    start.record()
    # outputs = model(**inputs)
    outputs = model(input_values)
    end.record()